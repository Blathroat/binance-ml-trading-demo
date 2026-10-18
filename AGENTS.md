# Repository Instructions

## Project Shape
- This is a small Python course demo for Binance spot OHLCV data, feature engineering, rule signals, supervised classification, and a Gradio UI.
- Main entrypoints are `demo/binance_spot_demo.py`, `demo/binance_ml_demo.py`, and `demo/gradio_app.py`.
- `demo/binance_ml_demo.py` imports feature/data helpers from `demo/binance_spot_demo.py`; keep shared OHLCV loading and feature changes there when possible.
- There is no package manifest beyond `requirements.txt`, and no checked-in test, lint, typecheck, or CI config.

## Commands
- Install dependencies with `pip install -r requirements.txt`.
- Run the spot feature/signal demo with `python .\demo\binance_spot_demo.py`.
- Run the ML demo offline from cached raw data with `python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 15m --raw-dir .\data\raw`.
- Fetch fresh Binance data for ML training with `python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 15m --days-back 180 --limit 8000 --fetch-online`.
- Use a specific raw CSV with `python .\demo\binance_ml_demo.py --raw-file .\outputs\raw_BTC_USDT_15m_20260525_102618.csv`.
- Start the UI with `python .\demo\gradio_app.py`; it binds to `127.0.0.1:7860` and allows file access only under `outputs/` and `data/`.

## Data And Outputs
- Default `data/raw` and `outputs` paths are anchored to the repository root, not the current working directory; do not use legacy `demo/data` or `demo/outputs` for new runs.
- `binance_ml_demo.py` defaults to local data: it reads the latest `raw_{symbol}_{timeframe}_*.csv` from root `data/raw` then root `outputs`; add `--fetch-online` to contact Binance.
- Online fetches save raw CSVs to root `data/raw/`; generated run artifacts go under root `outputs/`, with models under `outputs/models/*.joblib`.
- `.gitignore` excludes generated CSVs in `outputs/` and `data/`, but intentionally allows `outputs/models/*.joblib`.
- ML runs prune old stamped files in `outputs/` and `outputs/models/` according to `--keep-recent-runs` unless that value is `<= 0`.

## Modeling Notes
- The target is `target_up`, defined from `future_return_horizon > target_threshold`; default horizon is `1` K line and default threshold is `0.0015`.
- Splits are chronological, not shuffled: default train/validation/test ratios are `0.7 / 0.15 / 0.15`.
- Available fixed model names include `auto`, `随机森林`, `极端随机树`, `HistGradientBoosting`, `XGBoost_CPU`, and `XGBoost_GPU`.
- `xgboost` is optional at import time but listed in `requirements.txt`; GPU mode requires a working XGBoost GPU setup.

## Verification
- With no test suite, use focused smoke runs after edits: run the relevant script with a small `--limit` and cached `--raw-file` when available to avoid unnecessary network calls.
- For UI changes, verify `python .\demo\gradio_app.py` starts and the training/inference callbacks still map to the expected outputs.

## Documentation
- `README.md` is the primary user-facing run guide.
- `技术规划文档.md` and `模型调优实验报告.md` are course/report context; do not treat historical tuning results as guaranteed future performance.
