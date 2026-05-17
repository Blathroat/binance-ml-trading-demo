from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.binance_spot_demo import (
    add_features,
    fetch_ohlcv_data,
    load_local_raw_data,
    make_exchange,
    resolve_latest_raw_file,
)


@dataclass
class MLConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    days_back: int = 30
    exchange_id: str = "binance"
    output_dir: Path = Path("outputs")
    raw_dir: Path = Path("data/raw")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    fetch_online: bool = False
    raw_file: str | None = None


NUMERIC_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1",
    "log_return_1",
    "range_pct",
    "ma_7",
    "ma_21",
    "volatility_7",
    "volume_ma_7",
    "rsi_14",
    "close_vs_ma7",
    "close_vs_ma21",
]

CATEGORICAL_FEATURES = ["symbol", "exchange", "timeframe", "day_of_week", "hour", "trend_state"]


def make_one_hot_encoder():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def save_raw_input(raw_df: pd.DataFrame, config: MLConfig) -> Path:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = config.raw_dir / f"raw_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    return raw_path


def load_or_fetch_raw_data(config: MLConfig) -> tuple[pd.DataFrame, str, Path]:
    if config.fetch_online:
        try:
            exchange = make_exchange(config.exchange_id)
            raw_df = fetch_ohlcv_data(exchange, config.symbol, config.timeframe, config.days_back, config.limit)
            raw_path = save_raw_input(raw_df, config)
            return raw_df, "online", raw_path
        except Exception as exc:
            print(f"[警告] 联网抓取失败，改为读取本地原始数据：{exc}")

    raw_path = resolve_latest_raw_file(config.symbol, config.timeframe, search_dirs=(config.raw_dir, Path("outputs")), raw_file=config.raw_file)
    raw_df = load_local_raw_data(raw_path, symbol=config.symbol, timeframe=config.timeframe, exchange_id=config.exchange_id)
    return raw_df, "local", raw_path


def build_dataset(config: MLConfig) -> tuple[pd.DataFrame, str, Path]:
    raw_df, data_source, raw_path = load_or_fetch_raw_data(config)
    feature_df = add_features(raw_df)
    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)
    feature_df["future_close"] = feature_df["close"].shift(-1)
    feature_df["future_return_1"] = feature_df["future_close"] / feature_df["close"] - 1
    feature_df["target_up"] = (feature_df["future_return_1"] > 0).astype(int)
    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df, data_source, raw_path


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0~1 之间")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须在 0~1 之间")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    n = len(df)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) < 20 or len(val_df) < 5 or len(test_df) < 5:
        raise ValueError("数据量太少，无法稳定切分训练集、验证集和测试集")

    return train_df, val_df, test_df


def build_pipeline(model):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def safe_auc(y_true, proba):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return float("nan")


def evaluate_model(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_auc(y, proba) if proba is not None else float("nan"),
    }
    return metrics


def metrics_row(model_name: str, split_name: str, metrics: dict) -> dict:
    row = {"model": model_name, "split": split_name}
    row.update(metrics)
    return row


def choose_best_model(results: list[dict]) -> str:
    val_rows = [row for row in results if row["split"] == "验证集"]
    return max(val_rows, key=lambda r: (r["f1"], r["accuracy"]))["model"]


def save_outputs(df: pd.DataFrame, pred_df: pd.DataFrame, metrics_df: pd.DataFrame, config: MLConfig) -> tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pred_path = config.output_dir / f"ml_test_predictions_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    metrics_path = config.output_dir / f"ml_metrics_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary_path = config.output_dir / f"ml_summary_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.json"
    summary = {
        "rows": int(len(df)),
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "days_back": config.days_back,
        "fetch_online": config.fetch_online,
        "raw_file": config.raw_file,
        "best_model": pred_df.attrs.get("best_model", "unknown"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pred_path, metrics_path


def run_demo(config: MLConfig) -> None:
    df, data_source, raw_path = build_dataset(config)
    train_df, val_df, test_df = time_split(df, config.train_ratio, config.val_ratio)

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df["target_up"]
    X_val, y_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], val_df["target_up"]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df["target_up"]

    models = {
        "逻辑回归": build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        "随机森林": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    results: list[dict] = []
    fitted_models = {}

    print("开始训练涨跌预测模型...")
    print(f"数据来源：{data_source} | 原始文件：{raw_path}")
    print(f"样本总数：{len(df)} | 训练集：{len(train_df)} | 验证集：{len(val_df)} | 测试集：{len(test_df)}")

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        train_metrics = evaluate_model(pipeline, X_train, y_train)
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        results.append(metrics_row(name, "训练集", train_metrics))
        results.append(metrics_row(name, "验证集", val_metrics))
        print(
            f"{name} | 训练集 F1={train_metrics['f1']:.4f} | 验证集 F1={val_metrics['f1']:.4f} | 验证集 AUC={val_metrics['roc_auc']:.4f}"
        )

    metrics_df = pd.DataFrame(results)
    best_model_name = choose_best_model(results)
    best_model = fitted_models[best_model_name]

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_train_val, y_train_val)

    test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"最优模型：{best_model_name}")
    print(
        f"测试集结果 | Accuracy={test_metrics['accuracy']:.4f} | Precision={test_metrics['precision']:.4f} | Recall={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f} | AUC={test_metrics['roc_auc']:.4f}"
    )

    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    pred_df = test_df[["timestamp", "symbol", "exchange", "timeframe", "close", "future_close", "future_return_1", "target_up"]].copy()
    pred_df["pred_target_up"] = test_pred
    pred_df["pred_proba_up"] = test_proba
    pred_df.attrs["best_model"] = best_model_name

    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame([metrics_row(best_model_name, "测试集", test_metrics)]),
        ],
        ignore_index=True,
    )

    pred_path, metrics_path = save_outputs(df, pred_df, metrics_df, config)

    print(f"测试集预测结果已保存：{pred_path}")
    print(f"评估指标已保存：{metrics_path}")
    print("说明：目标变量为下一根 K 线是否上涨（target_up），仅用于课程演示与研究。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 ccxt + Binance 的涨跌预测 Demo")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易对，例如 BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="K线周期，例如 1h、15m、4h")
    parser.add_argument("--limit", type=int, default=500, help="最多抓取的 K 线条数")
    parser.add_argument("--days-back", type=int, default=30, help="回看最近多少天的数据")
    parser.add_argument("--exchange", default="binance", help="ccxt 支持的交易所 id")
    parser.add_argument("--output-dir", default="outputs", help="结果保存目录")
    parser.add_argument("--raw-dir", default="data/raw", help="原始数据目录，默认从这里读取最新 raw 文件")
    parser.add_argument("--raw-file", default=None, help="手动指定原始数据 CSV 文件路径")
    parser.add_argument("--fetch-online", action="store_true", help="联网抓取 Binance 数据并自动保存到本地 raw 目录")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = MLConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        days_back=args.days_back,
        exchange_id=args.exchange,
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        fetch_online=args.fetch_online,
        raw_file=args.raw_file,
    )
    run_demo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

