"""Command line: python -m btc_forecast run [--seeds 5] [--epochs 30] [--no-lstm] [--download]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import benchmark, data, plots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="btc_forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the walk-forward benchmark")
    run.add_argument("--seeds", type=int, default=5)
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--units", type=int, default=32)
    run.add_argument("--no-lstm", action="store_true")
    run.add_argument("--download", action="store_true", help="fetch fresh data with yfinance instead of the snapshot")
    run.add_argument("--data", default=None, help="path to a Date,Open,High,Low,Close,Volume CSV")
    run.add_argument("--out", default="results")
    run.add_argument("--docs", default="docs", help="where figures go ('' to skip)")
    a = ap.parse_args(argv)

    df = data.download() if a.download else data.load_snapshot(a.data)
    res = benchmark.run_benchmark(df, seeds=a.seeds, epochs=a.epochs, units=a.units, use_lstm=not a.no_lstm,
                                  log=print)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot = res.pop("_plot")
    (out / "btc.json").write_text(json.dumps(res, indent=2, default=float))
    md = benchmark.to_markdown(res)
    (out / "btc.md").write_text(md)
    print(md)
    if a.docs:
        d = Path(a.docs)
        d.mkdir(parents=True, exist_ok=True)
        res["_plot"] = plot
        plots.plot_last_fold(res, str(d / "last_fold.png"))
        plots.plot_ratio_bars(benchmark.aggregate(res), str(d / "rmse_ratio.png"))
    print(f"runtime: {res['runtime_s']:.0f} s")
    return 0
