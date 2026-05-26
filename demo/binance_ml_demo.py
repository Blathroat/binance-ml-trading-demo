from __future__ import annotations

# pyright: reportGeneralTypeIssues=false

import argparse
from importlib import import_module
import json
import os
import re
import pickle
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    _xgb_module = import_module("xgboost")
    XGBClassifier = getattr(_xgb_module, "XGBClassifier", None)
except Exception:  # optional dependency
    XGBClassifier = None

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
from demo.model_store import load_model_artifact, save_model_artifact


@dataclass
class MLConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    limit: int = 8000
    days_back: int = 180
    exchange_id: str = "binance"
    output_dir: Path = Path("outputs")
    raw_dir: Path = Path("data/raw")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    fetch_online: bool = False
    raw_file: str | None = None
    target_horizon: int = 1
    target_threshold: float = 0.0015
    fee_rate: float = 0.001
    use_gpu: bool = False
    gpu_backend: str = "auto"
    fixed_model_name: str = "auto"
    selection_metric: str = "total_return"
    min_trades: int = 5
    keep_recent_runs: int = 20


@dataclass
class InferenceConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    limit: int = 8000
    days_back: int = 180
    exchange_id: str = "binance"
    output_dir: Path = Path("outputs")
    raw_dir: Path = Path("data/raw")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    fetch_online: bool = False
    raw_file: str | None = None
    target_horizon: int = 1
    target_threshold: float = 0.0015
    fee_rate: float = 0.001
    keep_recent_runs: int = 20


@dataclass
class TrainingResult:
    config: MLConfig
    data_source: str
    raw_path: Path
    feature_path: Path
    model_path: Path
    pred_path: Path
    metrics_path: Path
    summary_path: Path
    best_model_name: str
    best_threshold: float
    validation_backtest_metrics: dict
    test_backtest_metrics: dict
    test_backtest_detail: pd.DataFrame
    metrics_df: pd.DataFrame
    pred_df: pd.DataFrame
    feature_df: pd.DataFrame
    test_metrics: dict
    train_rows: int
    val_rows: int
    test_rows: int
    class_balance: dict[str, int]
    deleted_output_files: list[Path]


NUMERIC_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1",
    "return_3",
    "return_6",
    "log_return_1",
    "momentum_12",
    "volume_change_1",
    "volume_change_3",
    "range_pct",
    "ma_5",
    "ma_13",
    "ma_7",
    "ma_21",
    "ma_34",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_7",
    "volatility_14",
    "volume_ma_7",
    "volume_ma_21",
    "volume_zscore_20",
    "rsi_7",
    "rsi_14",
    "atr_14",
    "bb_width",
    "price_position_14",
    "close_vs_ma5",
    "close_vs_ma13",
    "close_vs_ma7",
    "close_vs_ma21",
    "close_vs_ema12",
    "close_vs_ema26",
    "trend_strength",
]

CATEGORICAL_FEATURES = ["symbol", "exchange", "timeframe", "day_of_week", "hour", "minute", "trend_state"]


def make_one_hot_encoder():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


OUTPUT_STAMP_RE = re.compile(r"_(\d{8}_\d{6})(?=\.[^.]+$)")


def extract_output_stamp(path: Path) -> str | None:
    match = OUTPUT_STAMP_RE.search(path.name)
    return match.group(1) if match else None


def prune_old_outputs(output_dir: Path, keep_recent_runs: int) -> list[Path]:
    if keep_recent_runs <= 0 or not output_dir.exists():
        return []

    tracked_dirs = {output_dir.resolve(), (output_dir / "models").resolve()}
    stamped_files: list[tuple[str, Path]] = []

    for parent in (output_dir, output_dir / "models"):
        if not parent.exists() or not parent.is_dir():
            continue
        for path in parent.iterdir():
            if path.is_file():
                stamp = extract_output_stamp(path)
                if stamp:
                    stamped_files.append((stamp, path))

    if not stamped_files:
        return []

    unique_stamps = sorted({stamp for stamp, _ in stamped_files}, reverse=True)
    keep_stamps = set(unique_stamps[:keep_recent_runs])
    deleted: list[Path] = []
    for stamp, path in stamped_files:
        if stamp not in keep_stamps:
            try:
                path.unlink()
                deleted.append(path)
            except FileNotFoundError:
                pass

    for directory in (output_dir / "models", output_dir):
        if directory.exists() and directory.is_dir() and directory.resolve() in tracked_dirs:
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass

    return deleted


def build_model_path(config: MLConfig, stamp: str) -> Path:
    model_dir = config.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = config.symbol.replace("/", "_")
    return model_dir / f"model_{safe_symbol}_{config.timeframe}_{stamp}.joblib"


def load_trained_model(model_path: str | Path) -> dict[str, Any]:
    return load_model_artifact(model_path)


def resolve_model_feature_columns(artifact: dict[str, Any], df: pd.DataFrame) -> list[str]:
    feature_columns = artifact.get("feature_columns")
    if isinstance(feature_columns, (list, tuple)) and feature_columns:
        normalized = [str(col) for col in feature_columns]
        missing = [col for col in normalized if col not in df.columns]
        if missing:
            raise ValueError(f"模型所需特征列缺失：{missing}")
        return normalized
    fallback_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in fallback_columns if col not in df.columns]
    if missing:
        raise ValueError(f"当前数据缺少建模所需特征列：{missing}")
    return fallback_columns


def run_loaded_model(config: MLConfig, model_path: str | Path, verbose: bool = True) -> TrainingResult:
    df, data_source, raw_path = build_dataset(config)
    train_df, val_df, test_df = time_split(df, config.train_ratio, config.val_ratio)
    artifact = load_trained_model(model_path)
    model = artifact["model"]
    model_path = Path(model_path)
    config.fixed_model_name = f"joblib:{model_path.name}"

    if not hasattr(model, "predict_proba"):
        raise ValueError("当前 joblib 模型不支持 predict_proba，无法在网页中展示概率和回测结果")

    feature_columns = resolve_model_feature_columns(artifact, df)
    X_train, y_train = train_df[feature_columns], train_df["target_up"]
    X_val, y_val = val_df[feature_columns], val_df["target_up"]
    X_test, y_test = test_df[feature_columns], test_df["target_up"]

    saved_config = artifact.get("config", {}) if isinstance(artifact.get("config"), dict) else {}
    model_name = str(artifact.get("best_model_name") or model_path.stem)
    best_threshold = float(artifact.get("best_threshold", 0.5))

    if verbose:
        print("开始加载已有 joblib 模型进行评估...")
        print(f"模型文件：{model_path}")
        if saved_config:
            saved_symbol = saved_config.get("symbol")
            saved_timeframe = saved_config.get("timeframe")
            if saved_symbol and saved_timeframe and (saved_symbol != config.symbol or saved_timeframe != config.timeframe):
                print(
                    f"[警告] 当前选择的 symbol/timeframe 为 {config.symbol} / {config.timeframe}，与模型保存时的 {saved_symbol} / {saved_timeframe} 不一致"
                )
        print(f"数据来源：{data_source} | 原始文件：{raw_path}")
        print(f"样本总数：{len(df)} | 训练集：{len(train_df)} | 验证集：{len(val_df)} | 测试集：{len(test_df)}")

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    val_pred = (val_prob >= best_threshold).astype(int)
    tuned_test_pred = (test_prob >= best_threshold).astype(int)

    validation_backtest_metrics, validation_backtest_detail = simulate_long_only_backtest(
        val_df["future_return_horizon"],
        val_prob,
        best_threshold,
        config.fee_rate,
        config.timeframe,
    )
    test_backtest_metrics, test_backtest_df = simulate_long_only_backtest(
        test_df["future_return_horizon"],
        test_prob,
        best_threshold,
        config.fee_rate,
        config.timeframe,
    )
    tuned_test_metrics = {
        "accuracy": float(accuracy_score(y_test, tuned_test_pred)),
        "precision": float(precision_score(y_test, tuned_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, tuned_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, tuned_test_pred, zero_division=0)),
        "roc_auc": safe_auc(y_test, test_prob),
    }

    results: list[dict] = [
        metrics_row(model_name, "训练集", train_metrics),
        metrics_row(model_name, "验证集", val_metrics),
        metrics_row(model_name, "验证集_回测", validation_backtest_metrics),
        metrics_row(model_name, "测试集", test_metrics),
        {"model": model_name, "split": "测试集_阈值调优", **tuned_test_metrics, "best_threshold": best_threshold},
        {"model": model_name, "split": "测试集_回测", **test_backtest_metrics, "best_threshold": best_threshold},
    ]
    metrics_df = pd.DataFrame(results)

    pred_df = test_df[
        ["timestamp", "symbol", "exchange", "timeframe", "close", "future_close", "future_return_horizon", "target_up"]
    ].copy()
    pred_df["pred_target_up"] = tuned_test_pred
    pred_df["pred_proba_up"] = test_prob
    pred_df = pd.concat([pred_df.reset_index(drop=True), test_backtest_df.reset_index(drop=True)], axis=1)

    class_balance = {str(k): int(v) for k, v in df["target_up"].value_counts().sort_index().to_dict().items()}
    feature_path, pred_path, metrics_path, summary_path = save_outputs(
        df,
        pred_df,
        metrics_df,
        config,
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        model_path=Path(model_path),
        best_model_name=model_name,
        best_threshold=best_threshold,
        validation_backtest_metrics=validation_backtest_metrics,
        test_backtest_metrics=test_backtest_metrics,
        data_source=data_source,
        raw_path=raw_path,
        class_balance=class_balance,
        keep_recent_runs=config.keep_recent_runs,
    )

    deleted_output_files = prune_old_outputs(config.output_dir, config.keep_recent_runs)

    if verbose:
        print(f"验证集回测结果：{validation_backtest_metrics}")
        print(f"测试集回测结果：{test_backtest_metrics}")
        print(f"特征数据已保存：{feature_path}")
        print(f"测试集预测结果已保存：{pred_path}")
        print(f"评估指标已保存：{metrics_path}")
        print(f"汇总文件已保存：{summary_path}")
        if deleted_output_files:
            print(f"已清理旧输出文件：{len(deleted_output_files)} 个")

    return TrainingResult(
        config=config,
        data_source=data_source,
        raw_path=raw_path,
        feature_path=feature_path,
        model_path=Path(model_path),
        pred_path=pred_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        best_model_name=model_name,
        best_threshold=best_threshold,
        validation_backtest_metrics=validation_backtest_metrics,
        test_backtest_metrics=test_backtest_metrics,
        test_backtest_detail=test_backtest_df,
        metrics_df=metrics_df,
        pred_df=pred_df,
        feature_df=df,
        test_metrics=tuned_test_metrics,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        class_balance=class_balance,
        deleted_output_files=deleted_output_files,
    )


def run_inference(config: InferenceConfig, model_path: str | Path, verbose: bool = True) -> TrainingResult:
    train_config = MLConfig(
        symbol=config.symbol,
        timeframe=config.timeframe,
        limit=config.limit,
        days_back=config.days_back,
        exchange_id=config.exchange_id,
        output_dir=config.output_dir,
        raw_dir=config.raw_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        fetch_online=config.fetch_online,
        raw_file=config.raw_file,
        target_horizon=config.target_horizon,
        target_threshold=config.target_threshold,
        fee_rate=config.fee_rate,
        keep_recent_runs=config.keep_recent_runs,
    )
    return run_loaded_model(train_config, model_path, verbose=verbose)


def run_batch_training(config: MLConfig, timeframes: list[str], verbose: bool = True) -> tuple[list[TrainingResult], pd.DataFrame, TrainingResult]:
    if not timeframes:
        raise ValueError("请至少选择一个分时周期进行批量训练")

    results: list[TrainingResult] = []
    rows: list[dict[str, Any]] = []
    for timeframe in timeframes:
        tf_config = replace(config, timeframe=timeframe)
        result = run_pipeline(tf_config, verbose=verbose)
        results.append(result)
        rows.append(
            {
                "symbol": result.config.symbol,
                "timeframe": result.config.timeframe,
                "best_model": result.best_model_name,
                "best_threshold": result.best_threshold,
                "test_total_return": float(result.test_backtest_metrics.get("total_return", float("nan"))),
                "test_annual_return": float(result.test_backtest_metrics.get("annual_return", float("nan"))),
                "test_sharpe": float(result.test_backtest_metrics.get("sharpe", float("nan"))),
                "test_max_drawdown": float(result.test_backtest_metrics.get("max_drawdown", float("nan"))),
                "model_path": str(result.model_path),
                "summary_path": str(result.summary_path),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["test_total_return", "test_sharpe", "test_annual_return"],
        ascending=False,
        na_position="last",
    )
    best_row = summary_df.iloc[0]
    best_result = next(
        result
        for result in results
        if result.config.timeframe == best_row["timeframe"] and result.config.symbol == best_row["symbol"]
    )
    return results, summary_df, best_result


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
    feature_df["future_close"] = feature_df["close"].shift(-config.target_horizon)
    feature_df["future_return_horizon"] = feature_df["future_close"] / feature_df["close"] - 1
    feature_df["target_up"] = (feature_df["future_return_horizon"] > config.target_threshold).astype(int)
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
    return max(val_rows, key=lambda r: (r.get("validation_score", float("-inf")), r["accuracy"], r["f1"]))["model"]


def find_best_threshold(
    returns: pd.Series,
    prob: np.ndarray,
    fee_rate: float,
    timeframe: str,
    min_trades: int,
    selection_metric: str,
) -> tuple[float, float, dict]:
    best_thr = 0.5
    best_score = float("-inf")
    best_metrics: dict = {}
    for thr in np.linspace(0.05, 0.95, 181):
        backtest_metrics, _ = simulate_long_only_backtest(returns, prob, float(thr), fee_rate, timeframe)
        score = score_backtest(backtest_metrics, min_trades, selection_metric)
        if score > best_score or (score == best_score and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = float(score)
            best_thr = float(thr)
            best_metrics = backtest_metrics
    return best_thr, best_score, best_metrics


def timeframe_to_periods_per_year(timeframe: str) -> int:
    mapping = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}
    return mapping.get(timeframe, 365 * 24)


def simulate_long_only_backtest(
    returns: pd.Series,
    prob: np.ndarray,
    threshold: float,
    fee_rate: float,
    timeframe: str,
) -> tuple[dict, pd.DataFrame]:
    returns_np = pd.to_numeric(returns, errors="coerce").astype(float).to_numpy(dtype=float)
    prob = np.asarray(prob, dtype=float)
    position_np = (prob >= threshold).astype(np.int8)
    turnover_np = np.abs(np.diff(np.r_[0, position_np])).astype(float)
    net_return_np = position_np.astype(float) * returns_np - turnover_np * fee_rate
    equity_np = np.cumprod(1.0 + net_return_np)
    running_max_np = np.maximum.accumulate(equity_np)
    drawdown_np = equity_np / running_max_np - 1.0
    periods_per_year = timeframe_to_periods_per_year(timeframe)
    std = float(net_return_np.std(ddof=0))
    sharpe = float(net_return_np.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    active_mask = position_np.astype(bool)
    win_rate = float((net_return_np[active_mask] > 0).mean()) if active_mask.any() else 0.0
    trade_count = int(turnover_np.sum())
    total_return = float(equity_np[-1] - 1.0) if len(equity_np) else 0.0
    max_drawdown = float(drawdown_np.min()) if len(drawdown_np) else 0.0
    fee_paid = float(turnover_np.sum() * fee_rate)
    metrics = {
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (periods_per_year / max(len(net_return_np), 1)) - 1.0) if len(net_return_np) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "fee_paid": fee_paid,
        "exposure": float(position_np.mean()) if len(position_np) else 0.0,
    }
    detail = pd.DataFrame(
        {
            "position": position_np,
            "turnover": turnover_np,
            "gross_return": position_np.astype(float) * returns_np,
            "net_return": net_return_np,
            "equity_curve": equity_np,
            "drawdown": drawdown_np,
        }
    )
    return metrics, detail


def score_backtest(metrics: dict, min_trades: int, selection_metric: str) -> float:
    if metrics["trade_count"] < min_trades:
        return float("-inf")
    primary = float(metrics.get(selection_metric, float("nan")))
    if np.isnan(primary):
        return float("-inf")
    return primary + 0.05 * float(metrics.get("sharpe", 0.0)) + 0.5 * float(metrics.get("total_return", 0.0)) - 0.1 * abs(float(metrics.get("max_drawdown", 0.0)))


def make_xgb_pipeline(params: dict, use_gpu: bool) -> object | None:
    xgb_cls = XGBClassifier
    if xgb_cls is None:
        return None
    xgb_cls = cast(Any, xgb_cls)
    xgb_kwargs = dict(params)
    xgb_kwargs.update(tree_method="hist", device="cuda" if use_gpu else "cpu")
    return build_pipeline(xgb_cls(**xgb_kwargs))


def build_candidate_models(config: MLConfig):
    cpu_threads = max(1, (os.cpu_count() or 1) - 1)
    logistic = build_pipeline(
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.8, random_state=42, solver="saga")
    )
    random_forest = build_pipeline(
        RandomForestClassifier(
            n_estimators=800,
            max_depth=14,
            min_samples_leaf=2,
            min_samples_split=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=cpu_threads,
        )
    )
    extra_trees = build_pipeline(
        ExtraTreesClassifier(
            n_estimators=900,
            max_depth=16,
            min_samples_leaf=2,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=cpu_threads,
        )
    )
    hist_gbdt = build_pipeline(
        HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.03,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=0.8,
            random_state=42,
        )
    )
    voting = VotingClassifier(
        estimators=[
            ("lr", logistic),
            ("rf", random_forest),
            ("et", extra_trees),
        ],
        voting="soft",
        n_jobs=cpu_threads,
        weights=[1, 2, 2],
    )

    models: dict[str, Any] = {}
    models["逻辑回归"] = logistic
    models["随机森林"] = random_forest
    models["极端随机树"] = extra_trees
    models["HistGradientBoosting"] = hist_gbdt
    models["软投票集成"] = voting

    if XGBClassifier is not None:
        xgb_common = dict(
            n_estimators=900,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=cpu_threads,
        )
        xgb_cpu = make_xgb_pipeline(xgb_common, use_gpu=False)
        if xgb_cpu is not None:
            models["XGBoost_CPU"] = xgb_cpu
        if config.use_gpu and config.gpu_backend != "none":
            xgb_gpu = make_xgb_pipeline(xgb_common, use_gpu=True)
            if xgb_gpu is not None:
                models["XGBoost_GPU"] = xgb_gpu

    if config.fixed_model_name != "auto":
        if config.fixed_model_name not in models:
            available = ", ".join(sorted(models.keys()))
            raise ValueError(f"不支持的模型选择：{config.fixed_model_name}。可选：auto 或 {available}")
        return {config.fixed_model_name: models[config.fixed_model_name]}

    return models


def save_outputs(
    feature_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: MLConfig,
    *,
    stamp: str,
    model_path: Path,
    best_model_name: str,
    best_threshold: float,
    validation_backtest_metrics: dict,
    test_backtest_metrics: dict,
    data_source: str,
    raw_path: Path,
    class_balance: dict[str, int],
    keep_recent_runs: int,
) -> tuple[Path, Path, Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = config.output_dir / f"ml_features_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    pred_path = config.output_dir / f"ml_test_predictions_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    metrics_path = config.output_dir / f"ml_metrics_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    summary_path = config.output_dir / f"ml_summary_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.json"

    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(feature_df)),
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "days_back": config.days_back,
        "target_horizon": config.target_horizon,
        "target_threshold": config.target_threshold,
        "fetch_online": config.fetch_online,
        "raw_file": config.raw_file,
        "data_source": data_source,
        "raw_path": str(raw_path),
        "model_path": str(model_path),
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "validation_backtest": validation_backtest_metrics,
        "test_backtest": test_backtest_metrics,
        "class_balance": class_balance,
        "keep_recent_runs": keep_recent_runs,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return feature_path, pred_path, metrics_path, summary_path


def run_pipeline(config: MLConfig, verbose: bool = True) -> TrainingResult:
    df, data_source, raw_path = build_dataset(config)
    train_df, val_df, test_df = time_split(df, config.train_ratio, config.val_ratio)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df["target_up"]
    X_val, y_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], val_df["target_up"]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df["target_up"]

    models = build_candidate_models(config)

    results: list[dict] = []
    fitted_models = {}
    best_thresholds: dict[str, float] = {}
    validation_backtests: dict[str, dict] = {}
    validation_scores: dict[str, float] = {}

    if verbose:
        print("开始训练涨跌预测模型...")
        print(f"数据来源：{data_source} | 原始文件：{raw_path}")
        print(
            f"样本总数：{len(df)} | 训练集：{len(train_df)} | 验证集：{len(val_df)} | 测试集：{len(test_df)} | 预测窗口：未来 {config.target_horizon} 根 K 线"
        )
        class_counts = df["target_up"].value_counts().sort_index().to_dict()
        print(f"样本标签分布：{class_counts}")

    for name, pipeline in models.items():
        try:
            pipeline.fit(X_train, y_train)
            fitted_models[name] = pipeline
            train_metrics = evaluate_model(pipeline, X_train, y_train)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            best_thr, best_val_score, val_backtest_metrics = find_best_threshold(
                val_df["future_return_horizon"],
                val_prob,
                config.fee_rate,
                config.timeframe,
                config.min_trades,
                config.selection_metric,
            )
            best_thresholds[name] = best_thr
            val_pred = (val_prob >= best_thr).astype(int)
            validation_backtests[name] = val_backtest_metrics
            validation_scores[name] = score_backtest(val_backtest_metrics, config.min_trades, config.selection_metric)
            val_metrics = {
                "accuracy": float(accuracy_score(y_val, val_pred)),
                "precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "recall": float(recall_score(y_val, val_pred, zero_division=0)),
                "f1": float(f1_score(y_val, val_pred, zero_division=0)),
                "roc_auc": safe_auc(y_val, val_prob),
                "validation_score": float(validation_scores[name]),
                "val_total_return": float(val_backtest_metrics["total_return"]),
                "val_annual_return": float(val_backtest_metrics["annual_return"]),
                "val_sharpe": float(val_backtest_metrics["sharpe"]),
                "val_max_drawdown": float(val_backtest_metrics["max_drawdown"]),
                "val_trade_count": int(val_backtest_metrics["trade_count"]),
                "best_val_score": float(best_val_score),
            }
            val_metrics["best_threshold"] = float(best_thr)
            val_metrics["best_val_accuracy"] = float(val_metrics["accuracy"])
            results.append(metrics_row(name, "训练集", train_metrics))
            results.append(metrics_row(name, "验证集", val_metrics))
            results.append(metrics_row(name, "验证集_回测", val_backtest_metrics))
            if verbose:
                print(
                    f"{name} | 训练集 F1={train_metrics['f1']:.4f} | 验证集总收益={val_backtest_metrics['total_return']:.4f} | Sharpe={val_backtest_metrics['sharpe']:.2f} | 阈值={best_thr:.3f}"
                )
        except Exception as exc:
            if verbose:
                print(f"[警告] 模型 {name} 训练失败：{exc}")

    if not fitted_models:
        raise RuntimeError("所有候选模型都训练失败，请检查数据或特征工程设置")

    metrics_df = pd.DataFrame(results)
    best_model_name = choose_best_model(results)
    best_model = fitted_models[best_model_name]
    best_threshold = best_thresholds.get(best_model_name, 0.5)
    validation_backtest_metrics = validation_backtests.get(best_model_name, {})

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_train_val, y_train_val)

    test_metrics = evaluate_model(best_model, X_test, y_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    tuned_test_pred = (test_prob >= best_threshold).astype(int)
    test_backtest_metrics, test_backtest_df = simulate_long_only_backtest(
        test_df["future_return_horizon"],
        test_prob,
        best_threshold,
        config.fee_rate,
        config.timeframe,
    )
    tuned_test_metrics = {
        "accuracy": float(accuracy_score(y_test, tuned_test_pred)),
        "precision": float(precision_score(y_test, tuned_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, tuned_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, tuned_test_pred, zero_division=0)),
        "roc_auc": safe_auc(y_test, test_prob),
    }
    if verbose:
        print(f"最优模型：{best_model_name}")
        print(
            f"测试集结果 | Accuracy={tuned_test_metrics['accuracy']:.4f} | Precision={tuned_test_metrics['precision']:.4f} | Recall={tuned_test_metrics['recall']:.4f} | F1={tuned_test_metrics['f1']:.4f} | AUC={tuned_test_metrics['roc_auc']:.4f} | 总收益={test_backtest_metrics['total_return']:.4f} | Sharpe={test_backtest_metrics['sharpe']:.2f} | 阈值={best_threshold:.3f}"
        )

    test_pred = tuned_test_pred
    test_proba = test_prob
    pred_df = test_df[
        ["timestamp", "symbol", "exchange", "timeframe", "close", "future_close", "future_return_horizon", "target_up"]
    ].copy()
    pred_df["pred_target_up"] = test_pred
    pred_df["pred_proba_up"] = test_proba
    pred_df = pd.concat([pred_df.reset_index(drop=True), test_backtest_df.reset_index(drop=True)], axis=1)

    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame([metrics_row(best_model_name, "测试集", test_metrics)]),
            pd.DataFrame([{"model": best_model_name, "split": "测试集_阈值调优", **tuned_test_metrics, "best_threshold": best_threshold}]),
            pd.DataFrame([{"model": best_model_name, "split": "测试集_回测", **test_backtest_metrics, "best_threshold": best_threshold}]),
        ],
        ignore_index=True,
    )

    class_balance = {str(k): int(v) for k, v in df["target_up"].value_counts().sort_index().to_dict().items()}
    model_path = save_model_artifact(best_model, build_model_path(config, stamp))
    feature_path, pred_path, metrics_path, summary_path = save_outputs(
        df,
        pred_df,
        metrics_df,
        config,
        stamp=stamp,
        model_path=model_path,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        validation_backtest_metrics=validation_backtest_metrics,
        test_backtest_metrics=test_backtest_metrics,
        data_source=data_source,
        raw_path=raw_path,
        class_balance=class_balance,
        keep_recent_runs=config.keep_recent_runs,
    )

    deleted_output_files = prune_old_outputs(config.output_dir, config.keep_recent_runs)

    if verbose:
        print(f"特征数据已保存：{feature_path}")
        print(f"测试集预测结果已保存：{pred_path}")
        print(f"评估指标已保存：{metrics_path}")
        print(f"汇总文件已保存：{summary_path}")
        if deleted_output_files:
            print(f"已清理旧输出文件：{len(deleted_output_files)} 个")
        print("说明：目标变量为未来若干根 K 线是否上涨（target_up），仅用于课程演示与研究。")

    return TrainingResult(
        config=config,
        data_source=data_source,
        raw_path=raw_path,
        feature_path=feature_path,
        model_path=model_path,
        pred_path=pred_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        validation_backtest_metrics=validation_backtest_metrics,
        test_backtest_metrics=test_backtest_metrics,
        test_backtest_detail=test_backtest_df,
        metrics_df=metrics_df,
        pred_df=pred_df,
        feature_df=df,
        test_metrics=tuned_test_metrics,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        class_balance=class_balance,
        deleted_output_files=deleted_output_files,
    )


def run_demo(config: MLConfig) -> None:
    run_pipeline(config, verbose=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 ccxt + Binance 的涨跌预测 Demo")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易对，例如 BTC/USDT")
    parser.add_argument("--timeframe", default="15m", help="K线周期，例如 1h、15m、4h")
    parser.add_argument("--limit", type=int, default=8000, help="最多抓取的 K 线条数")
    parser.add_argument("--days-back", type=int, default=180, help="回看最近多少天的数据")
    parser.add_argument("--exchange", default="binance", help="ccxt 支持的交易所 id")
    parser.add_argument("--output-dir", default="outputs", help="结果保存目录")
    parser.add_argument("--raw-dir", default="data/raw", help="原始数据目录，默认从这里读取最新 raw 文件")
    parser.add_argument("--raw-file", default=None, help="手动指定原始数据 CSV 文件路径")
    parser.add_argument("--fetch-online", action="store_true", help="联网抓取 Binance 数据并自动保存到本地 raw 目录")
    parser.add_argument("--target-horizon", type=int, default=1, help="预测未来多少根 K 线的涨跌方向")
    parser.add_argument("--target-threshold", type=float, default=0.0015, help="涨跌标签阈值，过滤微小波动噪声")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="单边手续费比例，用于回测")
    parser.add_argument("--use-gpu", action="store_true", help="启用 GPU 训练选项（若安装了 xgboost）")
    parser.add_argument("--gpu-backend", default="auto", choices=["auto", "xgboost", "none"], help="GPU 后端选择")
    parser.add_argument("--fixed-model-name", default="auto", help="固定训练某个模型；auto 表示自动多模型对比")
    parser.add_argument("--selection-metric", default="total_return", choices=["total_return", "annual_return", "sharpe"], help="模型选择的主目标")
    parser.add_argument("--min-trades", type=int, default=5, help="验证集回测最少交易次数，否则视为无效")
    parser.add_argument("--keep-recent-runs", type=int, default=20, help="仅保留最近多少次运行的输出文件")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold,
        fee_rate=args.fee_rate,
        use_gpu=args.use_gpu,
        gpu_backend=args.gpu_backend,
        fixed_model_name=args.fixed_model_name,
        selection_metric=args.selection_metric,
        min_trades=args.min_trades,
        keep_recent_runs=args.keep_recent_runs,
    )
    run_demo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

