import json

import numpy as np

from btc_forecast import benchmark, cli


def test_benchmark_no_lstm(random_walk_df):
    res = benchmark.run_benchmark(random_walk_df, use_lstm=False, lookback=20, n_folds=3)
    names = [r["model"] for r in res["runs"]]
    assert names[0].startswith("Naive") and len(names) == 4
    naive = res["runs"][0]
    assert naive["pooled"]["rmse_ratio_vs_naive"] == 1.0
    assert len(res["folds"]) == 3
    assert "| Model |" in benchmark.to_markdown(res)


def test_benchmark_folds_do_not_overlap(random_walk_df):
    res = benchmark.run_benchmark(random_walk_df, use_lstm=False, lookback=20, n_folds=4)
    for a, b in zip(res["folds"], res["folds"][1:]):
        assert a["test_end"] < b["test_start"]


def test_cli_end_to_end(tmp_path, random_walk_df):
    csv = tmp_path / "d.csv"
    random_walk_df.to_csv(csv)
    rc = cli.main(["run", "--no-lstm", "--data", str(csv), "--out", str(tmp_path / "r"), "--docs", str(tmp_path / "d")])
    assert rc == 0
    out = json.loads((tmp_path / "r" / "btc.json").read_text())
    assert len(out["runs"]) == 4 and (tmp_path / "r" / "btc.md").exists()
    assert (tmp_path / "d" / "last_fold.png").exists() and (tmp_path / "d" / "rmse_ratio.png").exists()


def test_deterministic_models_have_zero_std(random_walk_df):
    rows = benchmark.aggregate(benchmark.run_benchmark(random_walk_df, use_lstm=False, lookback=20, n_folds=3))
    assert all(r["rmse_usd"][1] == 0 for r in rows)
    assert np.isnan(rows[0]["dir_acc"][0])  # naive never takes a side
