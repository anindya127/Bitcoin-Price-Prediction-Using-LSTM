Pooled over 5 walk-forward test folds (2021-01-06 to 2024-11-29, 1424 days). Mean ± std over 5 seeds for the LSTM; the other models are deterministic (std 0).

| Model | RMSE (USD) | MAE (USD) | RMSE ratio vs naive | Dir. accuracy (%) | DM p-value vs naive |
|---|---|---|---|---|---|
| Naive (random walk) | 1,439 ± 0 | 945 ± 0 | 1.000 ± 0.000 | n/a | - |
| Drift | 1,440 ± 0 | 947 ± 0 | 1.000 ± 0.000 | 49.9 ± 0.0 | 0.798 |
| Moving average (5d) | 1,563 ± 0 | 1,083 ± 0 | 1.086 ± 0.000 | 45.4 ± 0.0 | <0.001 |
| Ridge (lagged returns) | 1,454 ± 0 | 963 ± 0 | 1.010 ± 0.000 | 49.5 ± 0.0 | 0.121 |
| LSTM (32 units) | 1,438 ± 1 | 951 ± 1 | 0.999 ± 0.001 | 50.0 ± 0.2 | 0.806 |

Share of up-days in the test period (accuracy of 'always up'): 49.9%.

Per-fold test windows:

| Fold | Train days | Test days | Test period |
|---|---|---|---|
| 0 | 2136 | 284 | 2021-01-06 to 2021-10-16 |
| 1 | 2420 | 284 | 2021-10-17 to 2022-07-27 |
| 2 | 2704 | 284 | 2022-07-28 to 2023-05-07 |
| 3 | 2988 | 284 | 2023-05-08 to 2024-02-15 |
| 4 | 3272 | 288 | 2024-02-16 to 2024-11-29 |
