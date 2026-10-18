from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from binance_ml_demo import (
    CATEGORICAL_FEATURES,
    MLConfig,
    NUMERIC_FEATURES,
    time_split,
    simulate_long_only_backtest,
    find_best_threshold,
    score_backtest,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

OUT_DIR = ROOT_DIR.parent / "outputs" / "hyperparameter_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_CSV = ROOT_DIR.parent / "outputs" / "ml_features_BTC_USDT_4h_20260527_125712.csv"
CONFIG = MLConfig(symbol="BTC/USDT", timeframe="4h", target_horizon=6, target_threshold=0.0015)


def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, CATEGORICAL_FEATURES)])


def safe_auc(y_true, proba):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return float("nan")


def cv_backtest_score(estimator, X, y, returns, fee_rate=0.001):
    proba = estimator.predict_proba(X)[:, 1]
    backtest_metrics, _ = simulate_long_only_backtest(returns, proba, threshold=0.5, fee_rate=fee_rate, timeframe="4h")
    return backtest_metrics["total_return"]


def main():
    print("=" * 70)
    print("交叉验证 + 超参数搜索实验")
    print("=" * 70)
    print(f"数据: {FEATURE_CSV.name}")
    print(f"配置: BTC/USDT 4h, horizon=6, threshold=0.15%")

    print("\n[1/5] 加载数据...")
    df = pd.read_csv(FEATURE_CSV)
    train_df, val_df, test_df = time_split(df, CONFIG.train_ratio, CONFIG.val_ratio)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_all = pd.concat([train_df, val_df])[feature_cols]
    y_all = pd.concat([train_df, val_df])["target_up"]
    returns_all = pd.concat([train_df, val_df])["future_return_horizon"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]
    returns_test = test_df["future_return_horizon"]

    print(f"  训练+验证: {len(X_all)} 条, 测试: {len(X_test)} 条")
    print(f"  标签分布: 跌={int((y_all == 0).sum())} 涨={int((y_all == 1).sum())}")

    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\n  交叉验证策略: TimeSeriesSplit (5折)")
    print(f"  (时间序列数据不可随机打乱，使用滚动窗口切分)")

    cpu_threads = max(1, (os.cpu_count() or 1) - 1)

    results_all = {}

    print("\n" + "=" * 70)
    print("[2/5] 模型1: 逻辑回归 — GridSearchCV")
    print("=" * 70)

    lr_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, solver="lbfgs")),
    ])
    lr_param_grid = {
        "model__C": [0.01, 0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0],
    }
    lr_gs = GridSearchCV(
        lr_pipe, lr_param_grid, cv=tscv, scoring="f1",
        n_jobs=cpu_threads, verbose=1, refit=True,
    )
    lr_gs.fit(X_all, y_all)
    lr_best = lr_gs.best_estimator_
    lr_val_prob = lr_best.predict_proba(val_df[feature_cols])[:, 1]
    lr_val_returns = val_df["future_return_horizon"]
    lr_best_thr, _, _ = find_best_threshold(lr_val_returns, lr_val_prob, CONFIG.fee_rate, "4h", CONFIG.min_trades, CONFIG.selection_metric)
    lr_test_prob = lr_best.predict_proba(X_test)[:, 1]
    lr_bt, _ = simulate_long_only_backtest(returns_test, lr_test_prob, lr_best_thr, CONFIG.fee_rate, "4h")
    lr_eval = {
        "best_params": lr_gs.best_params_,
        "best_cv_f1": float(lr_gs.best_score_),
        "cv_mean_f1": float(lr_gs.cv_results_["mean_test_score"][lr_gs.best_index_]),
        "cv_std_f1": float(lr_gs.cv_results_["std_test_score"][lr_gs.best_index_]),
        "test_f1": float(f1_score(y_test, lr_best.predict(X_test), zero_division=0)),
        "test_auc": float(safe_auc(y_test, lr_test_prob)),
        "best_threshold": lr_best_thr,
        "test_total_return": float(lr_bt["total_return"]),
        "test_sharpe": float(lr_bt["sharpe"]),
        "test_max_drawdown": float(lr_bt["max_drawdown"]),
        "n_combinations": len(lr_gs.cv_results_["params"]),
    }
    results_all["逻辑回归"] = lr_eval
    print(f"  最优参数: {lr_gs.best_params_}")
    print(f"  CV F1: {lr_gs.best_score_:.4f} ± {lr_eval['cv_std_f1']:.4f}")
    print(f"  测试 F1: {lr_eval['test_f1']:.4f}  测试收益: {lr_eval['test_total_return']:+.4%}")

    print("\n" + "=" * 70)
    print("[3/5] 模型2: 随机森林 — GridSearchCV")
    print("=" * 70)

    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=cpu_threads)),
    ])
    rf_param_grid = {
        "model__n_estimators": [200, 400, 600, 800, 1000],
        "model__max_depth": [6, 10, 14, 18, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10, 20],
    }
    rf_gs = RandomizedSearchCV(
        rf_pipe, rf_param_grid, cv=tscv, scoring="f1",
        n_jobs=cpu_threads, verbose=1, refit=True, n_iter=80, random_state=42,
    )
    rf_gs.fit(X_all, y_all)
    rf_best = rf_gs.best_estimator_
    rf_val_prob = rf_best.predict_proba(val_df[feature_cols])[:, 1]
    rf_val_returns = val_df["future_return_horizon"]
    rf_best_thr, _, _ = find_best_threshold(rf_val_returns, rf_val_prob, CONFIG.fee_rate, "4h", CONFIG.min_trades, CONFIG.selection_metric)
    rf_test_prob = rf_best.predict_proba(X_test)[:, 1]
    rf_bt, _ = simulate_long_only_backtest(returns_test, rf_test_prob, rf_best_thr, CONFIG.fee_rate, "4h")
    rf_eval = {
        "best_params": rf_gs.best_params_,
        "best_cv_f1": float(rf_gs.best_score_),
        "cv_mean_f1": float(rf_gs.cv_results_["mean_test_score"][rf_gs.best_index_]),
        "cv_std_f1": float(rf_gs.cv_results_["std_test_score"][rf_gs.best_index_]),
        "test_f1": float(f1_score(y_test, rf_best.predict(X_test), zero_division=0)),
        "test_auc": float(safe_auc(y_test, rf_test_prob)),
        "best_threshold": rf_best_thr,
        "test_total_return": float(rf_bt["total_return"]),
        "test_sharpe": float(rf_bt["sharpe"]),
        "test_max_drawdown": float(rf_bt["max_drawdown"]),
        "n_combinations": len(rf_gs.cv_results_["params"]),
    }
    results_all["随机森林"] = rf_eval
    print(f"  最优参数: {rf_gs.best_params_}")
    print(f"  CV F1: {rf_gs.best_score_:.4f} ± {rf_eval['cv_std_f1']:.4f}")
    print(f"  测试 F1: {rf_eval['test_f1']:.4f}  测试收益: {rf_eval['test_total_return']:+.4%}")

    print("\n" + "=" * 70)
    print("[4/5] 模型3: SVM — GridSearchCV")
    print("=" * 70)

    svm_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
    ])
    svm_param_grid = {
        "model__C": [0.1, 0.5, 1.0, 5.0],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }
    svm_gs = GridSearchCV(
        svm_pipe, svm_param_grid, cv=tscv, scoring="f1",
        n_jobs=cpu_threads, verbose=1, refit=True,
    )
    svm_gs.fit(X_all, y_all)
    svm_best = svm_gs.best_estimator_
    svm_val_prob = svm_best.predict_proba(val_df[feature_cols])[:, 1]
    svm_val_returns = val_df["future_return_horizon"]
    svm_best_thr, _, _ = find_best_threshold(svm_val_returns, svm_val_prob, CONFIG.fee_rate, "4h", CONFIG.min_trades, CONFIG.selection_metric)
    svm_test_prob = svm_best.predict_proba(X_test)[:, 1]
    svm_bt, _ = simulate_long_only_backtest(returns_test, svm_test_prob, svm_best_thr, CONFIG.fee_rate, "4h")
    svm_eval = {
        "best_params": svm_gs.best_params_,
        "best_cv_f1": float(svm_gs.best_score_),
        "cv_mean_f1": float(svm_gs.cv_results_["mean_test_score"][svm_gs.best_index_]),
        "cv_std_f1": float(svm_gs.cv_results_["std_test_score"][svm_gs.best_index_]),
        "test_f1": float(f1_score(y_test, svm_best.predict(X_test), zero_division=0)),
        "test_auc": float(safe_auc(y_test, svm_test_prob)),
        "best_threshold": svm_best_thr,
        "test_total_return": float(svm_bt["total_return"]),
        "test_sharpe": float(svm_bt["sharpe"]),
        "test_max_drawdown": float(svm_bt["max_drawdown"]),
        "n_combinations": len(svm_gs.cv_results_["params"]),
    }
    results_all["SVM"] = svm_eval
    print(f"  最优参数: {svm_gs.best_params_}")
    print(f"  CV F1: {svm_gs.best_score_:.4f} ± {svm_eval['cv_std_f1']:.4f}")
    print(f"  测试 F1: {svm_eval['test_f1']:.4f}  测试收益: {svm_eval['test_total_return']:+.4%}")

    if XGBClassifier is not None:
        print("\n" + "=" * 70)
        print("[5/5] 模型4: XGBoost — GridSearchCV")
        print("=" * 70)

        xgb_pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=cpu_threads,
            )),
        ])
        xgb_param_grid = {
            "model__n_estimators": [200, 400, 600, 900],
            "model__max_depth": [3, 5, 6, 8, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.0, 0.5, 1.0, 2.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
        }
        xgb_gs = RandomizedSearchCV(
            xgb_pipe, xgb_param_grid, cv=tscv, scoring="f1",
            n_jobs=cpu_threads, verbose=1, refit=True, n_iter=100, random_state=42,
        )
        xgb_gs.fit(X_all, y_all)
        xgb_best = xgb_gs.best_estimator_
        xgb_val_prob = xgb_best.predict_proba(val_df[feature_cols])[:, 1]
        xgb_val_returns = val_df["future_return_horizon"]
        xgb_best_thr, _, _ = find_best_threshold(xgb_val_returns, xgb_val_prob, CONFIG.fee_rate, "4h", CONFIG.min_trades, CONFIG.selection_metric)
        xgb_test_prob = xgb_best.predict_proba(X_test)[:, 1]
        xgb_bt, _ = simulate_long_only_backtest(returns_test, xgb_test_prob, xgb_best_thr, CONFIG.fee_rate, "4h")
        xgb_eval = {
            "best_params": xgb_gs.best_params_,
            "best_cv_f1": float(xgb_gs.best_score_),
            "cv_mean_f1": float(xgb_gs.cv_results_["mean_test_score"][xgb_gs.best_index_]),
            "cv_std_f1": float(xgb_gs.cv_results_["std_test_score"][xgb_gs.best_index_]),
            "test_f1": float(f1_score(y_test, xgb_best.predict(X_test), zero_division=0)),
            "test_auc": float(safe_auc(y_test, xgb_test_prob)),
            "best_threshold": xgb_best_thr,
            "test_total_return": float(xgb_bt["total_return"]),
            "test_sharpe": float(xgb_bt["sharpe"]),
            "test_max_drawdown": float(xgb_bt["max_drawdown"]),
            "n_combinations": len(xgb_gs.cv_results_["params"]),
        }
        results_all["XGBoost"] = xgb_eval
        print(f"  最优参数: {xgb_gs.best_params_}")
        print(f"  CV F1: {xgb_gs.best_score_:.4f} ± {xgb_eval['cv_std_f1']:.4f}")
        print(f"  测试 F1: {xgb_eval['test_f1']:.4f}  测试收益: {xgb_eval['test_total_return']:+.4%}")

    print("\n" + "=" * 70)
    print("交叉验证 + 超参数搜索 结果汇总")
    print("=" * 70)

    print(f"\n{'模型':<12} {'CV F1':<12} {'测试F1':<10} {'测试AUC':<10} {'测试收益':<12} {'测试Sharpe':<12} {'搜索组合数':<10}")
    print("-" * 78)
    for name, r in results_all.items():
        print(f"{name:<12} {r['cv_mean_f1']:.4f}±{r['cv_std_f1']:.3f} {r['test_f1']:<10.4f} {r['test_auc']:<10.4f} {r['test_total_return']:>+10.2%} {r['test_sharpe']:>+10.2f} {r['n_combinations']:<10}")

    best_by_return = max(results_all.items(), key=lambda x: x[1]["test_total_return"])
    best_by_cv = max(results_all.items(), key=lambda x: x[1]["cv_mean_f1"])
    print(f"\n  回测收益最优: {best_by_return[0]} ({best_by_return[1]['test_total_return']:+.2%})")
    print(f"  CV F1 最优: {best_by_cv[0]} ({best_by_cv[1]['cv_mean_f1']:.4f})")

    print("\n生成对比图表...")

    models_names = list(results_all.keys())
    cv_f1s = [results_all[m]["cv_mean_f1"] for m in models_names]
    cv_stds = [results_all[m]["cv_std_f1"] for m in models_names]
    test_returns = [results_all[m]["test_total_return"] * 100 for m in models_names]
    test_sharpes = [results_all[m]["test_sharpe"] for m in models_names]
    test_f1s = [results_all[m]["test_f1"] for m in models_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    x = np.arange(len(models_names))
    width = 0.3
    colors_ret = ["#2ecc71" if r >= 0 else "#e74c3c" for r in test_returns]
    bars = ax1.bar(x, test_returns, width, color=colors_ret, edgecolor="white", linewidth=0.8)
    ax1.set_ylabel("测试总收益 (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, fontsize=11)
    ax1.set_title("交叉验证超参搜索后：各模型测试收益对比", fontsize=13, fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.8)
    for bar, ret in zip(bars, test_returns):
        y_pos = bar.get_height() + 0.5 if ret >= 0 else bar.get_height() - 2.5
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{ret:+.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    x2 = np.arange(len(models_names))
    bars2 = ax2.bar(x2 - width / 2, cv_f1s, width, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.6, label="CV F1 (均值)")
    ax2.errorbar(x2 - width / 2, cv_f1s, yerr=cv_stds, fmt="none", ecolor="black", capsize=5, linewidth=1.2)
    bars3 = ax2.bar(x2 + width / 2, test_f1s, width, color="#e74c3c", alpha=0.8, edgecolor="white", linewidth=0.6, label="测试集 F1")
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models_names, fontsize=11)
    ax2.set_title("交叉验证 F1 vs 测试集 F1 (含±1标准差)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, cv_f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold", rotation=90)
    for bar, val in zip(bars3, test_f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold", rotation=90)

    fig.tight_layout()
    path = OUT_DIR / "cv_hyperparameter_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [√] {path.name}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    x = np.arange(len(models_names))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, test_returns, width, color=["#2ecc71" if r >= 0 else "#e74c3c" for r in test_returns],
                    edgecolor="white", label="测试总收益 (%)")
    ax1.set_ylabel("测试总收益 (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, fontsize=10)
    ax1.axhline(y=0, color="black", linewidth=0.6)
    ax1.set_title("超参搜索后：测试收益 vs 测试 Sharpe", fontsize=13, fontweight="bold")
    for bar, ret in zip(bars1, test_returns):
        y_pos = bar.get_height() + 0.3 if ret >= 0 else bar.get_height() - 2
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{ret:+.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax2_b = ax1.twinx()
    ax2_b.plot(x + width / 2, test_sharpes, "s-", color="#8e44ad", linewidth=2, markersize=8, markerfacecolor="white",
               markeredgewidth=2, label="Sharpe")
    ax2_b.set_ylabel("Sharpe Ratio", fontsize=12, color="#8e44ad")
    for i, s in enumerate(test_sharpes):
        ax2_b.annotate(f"{s:.2f}", (x[i] + width / 2, s), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#8e44ad")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2_b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = axes[1]
    search_counts = [results_all[m]["n_combinations"] for m in models_names]
    bars = ax2.bar(models_names, search_counts, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"][:len(models_names)],
                   edgecolor="white", linewidth=0.8)
    ax2.set_ylabel("搜索组合数", fontsize=12)
    ax2.set_title("各模型超参数搜索空间大小", fontsize=13, fontweight="bold")
    for bar, cnt in zip(bars, search_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(cnt), ha="center", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "cv_hyperparameter_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [√] {path.name}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {
        "stamp": stamp,
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "target_horizon": 6,
        "target_threshold": 0.0015,
        "cv_strategy": "TimeSeriesSplit(n_splits=5)",
        "train_val_size": len(X_all),
        "test_size": len(X_test),
        "results": results_all,
    }
    json_path = OUT_DIR / f"cv_hyperparameter_results_{stamp}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [√] 结果已保存至 {json_path.name}")

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()