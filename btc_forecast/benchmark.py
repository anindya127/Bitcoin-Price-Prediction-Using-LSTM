"""Walk-forward benchmark: every model, every fold, several seeds for the stochastic ones."""
from __future__ import annotations

import time

import numpy as np

from . import metrics
from .data import build_dataset, walk_forward_folds
from .models import Drift, LSTMModel, MovingAverage, Naive, RidgeLags


def run_benchmark(df, seeds=5, epochs=30, units=32, use_lstm=True, lookback=60, n_folds=5,
                  initial_train_frac=0.6, log=None) -> dict:
    ds = build_dataset(df, lookback)
    folds = walk_forward_folds(len(ds.y), initial_train_frac, n_folds)
    t0 = time.time()
    specs = [(Naive(), 1), (Drift(), 1), (MovingAverage(5), 1), (RidgeLags(1.0), 1)]
    if use_lstm:
        specs += [(LSTMModel(units=units, epochs=epochs, seed=s), 1) for s in range(seeds)]
    runs = []  # one entry per (model, seed)
    preds = {}  # model name -> {seed: per-fold predicted returns}
    for model, _ in specs:
        seed = getattr(model, "seed", 0)
        fold_preds, fold_metrics, epochs_run = [], [], []
        for k, f in enumerate(folds):
            fit_idx = f.fit_idx if model.deterministic is False else f.train_idx
            model.fit(ds.X[fit_idx], ds.y[fit_idx], ds.X[f.val_idx], ds.y[f.val_idx])
            p = model.predict(ds.X[f.test_idx])
            fold_preds.append(p)
            fold_metrics.append(metrics.evaluate(ds.y[f.test_idx], p, ds.prev_close[f.test_idx],
                                                 ds.actual_close[f.test_idx]))
            if hasattr(model, "epochs_run_"):
                epochs_run.append(model.epochs_run_)
        all_idx = np.concatenate([f.test_idx for f in folds])
        pooled = metrics.evaluate(ds.y[all_idx], np.concatenate(fold_preds), ds.prev_close[all_idx],
                                  ds.actual_close[all_idx])
        runs.append({"model": model.name, "seed": seed, "deterministic": model.deterministic,
                     "pooled": pooled, "folds": fold_metrics, "epochs_run": epochs_run})
        preds.setdefault(model.name, {})[seed] = fold_preds
        if log:
            log(f"{model.name} seed={seed} RMSE={pooled['rmse_usd']:.1f}")
    all_idx = np.concatenate([f.test_idx for f in folds])
    naive_p = np.concatenate(preds[Naive.name][0])
    naive_rmse = next(r for r in runs if r["model"] == Naive.name)["pooled"]["rmse_usd"]
    for r in runs:
        r["pooled"]["rmse_ratio_vs_naive"] = metrics.rmse_ratio(r["pooled"]["rmse_usd"], naive_rmse)
        p = np.concatenate(preds[r["model"]][r["seed"]])
        pp = ds.prev_close[all_idx] * np.exp(p)
        npr = ds.prev_close[all_idx] * np.exp(naive_p)
        stat, pval = metrics.diebold_mariano(ds.actual_close[all_idx], pp, npr)
        r["pooled"]["dm_stat"], r["pooled"]["dm_p"] = stat, pval
        fr = []
        for k, f in enumerate(folds):
            nr = next(x for x in runs if x["model"] == Naive.name)["folds"][k]["rmse_usd"]
            fr.append(r["folds"][k]["rmse_usd"] / nr)
        r["fold_rmse_ratio"] = fr
    return {
        "config": {"seeds": seeds, "epochs": epochs, "units": units, "use_lstm": use_lstm, "lookback": lookback,
                   "n_folds": n_folds, "initial_train_frac": initial_train_frac, "val_frac": 0.1},
        "folds": [{"fold": k, "n_train": f.n_train, "n_test": len(f.test_idx),
                   "test_start": str(ds.dates[f.test_idx[0]].date()), "test_end": str(ds.dates[f.test_idx[-1]].date())}
                  for k, f in enumerate(folds)],
        "runs": runs,
        "runtime_s": time.time() - t0,
        "_plot": {"dates": ds.dates[folds[-1].test_idx], "actual": ds.actual_close[folds[-1].test_idx],
                  "prev": ds.prev_close[folds[-1].test_idx],
                  "preds": {m: v[min(v)][-1] for m, v in preds.items()}},
    }


def aggregate(result: dict) -> list[dict]:
    """Mean and std over seeds of the pooled metrics, per model (std is 0 for deterministic models)."""
    keys = ["rmse_usd", "mae_usd", "rmse_ret", "mae_ret", "dir_acc", "rmse_ratio_vs_naive", "dm_p"]
    out = []
    for name in dict.fromkeys(r["model"] for r in result["runs"]):
        rs = [r for r in result["runs"] if r["model"] == name]
        row = {"model": name, "n_seeds": len(rs), "deterministic": rs[0]["deterministic"],
               "up_share": rs[0]["pooled"]["up_share"], "n": rs[0]["pooled"]["n"]}
        for k in keys:
            v = np.array([r["pooled"][k] for r in rs], dtype=float)
            row[k] = (float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan"),
                      float(np.nanstd(v)) if not np.all(np.isnan(v)) else float("nan"))
        out.append(row)
    return out


def to_markdown(result: dict) -> str:
    rows = aggregate(result)
    c = result["config"]
    f = lambda t, d=0: "n/a (never predicts a side)" if np.isnan(t[0]) else f"{t[0]:,.{d}f} ± {t[1]:,.{d}f}"
    lines = [f"Pooled over {len(result['folds'])} walk-forward test folds "
             f"({result['folds'][0]['test_start']} to {result['folds'][-1]['test_end']}, {rows[0]['n']} days). "
             f"Mean ± std over {c['seeds']} seeds for the LSTM; the other models are deterministic (std 0).", "",
             "| Model | RMSE (USD) | MAE (USD) | RMSE ratio vs naive | Dir. accuracy (%) | DM p-value vs naive |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        da = "n/a" if np.isnan(r["dir_acc"][0]) else f"{100 * r['dir_acc'][0]:.1f} ± {100 * r['dir_acc'][1]:.1f}"
        dm = "-" if r["model"].startswith("Naive") else f"{r['dm_p'][0]:.3f}"
        lines.append(f"| {r['model']} | {f(r['rmse_usd'])} | {f(r['mae_usd'])} | {r['rmse_ratio_vs_naive'][0]:.3f} ± "
                     f"{r['rmse_ratio_vs_naive'][1]:.3f} | {da} | {dm} |")
    lines += ["", f"Share of up-days in the test period (accuracy of 'always up'): {100 * rows[0]['up_share']:.1f}%.",
              "", "Per-fold test windows:", "", "| Fold | Train days | Test days | Test period |", "|---|---|---|---|"]
    for fo in result["folds"]:
        lines.append(f"| {fo['fold']} | {fo['n_train']} | {fo['n_test']} | {fo['test_start']} to {fo['test_end']} |")
    return "\n".join(lines) + "\n"
