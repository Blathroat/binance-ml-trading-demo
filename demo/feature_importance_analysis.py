from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from binance_ml_demo import (
    CATEGORICAL_FEATURES,
    MLConfig,
    NUMERIC_FEATURES,
    time_split,
    load_trained_model,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_CN_MAP = {
    "open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价",
    "volume": "成交量", "return_1": "1周期收益率", "return_3": "3周期收益率",
    "return_6": "6周期收益率", "log_return_1": "对数收益率",
    "momentum_12": "12周期动量", "volume_change_1": "成交量变化(1)",
    "volume_change_3": "成交量变化(3)", "range_pct": "振幅百分比",
    "ma_5": "MA5", "ma_13": "MA13", "ma_7": "MA7", "ma_21": "MA21",
    "ma_34": "MA34", "ema_12": "EMA12", "ema_26": "EMA26",
    "macd": "MACD_DIF", "macd_signal": "MACD_DEA", "macd_hist": "MACD柱",
    "volatility_7": "波动率(7)", "volatility_14": "波动率(14)",
    "volume_ma_7": "成交量均线(7)", "volume_ma_21": "成交量均线(21)",
    "volume_zscore_20": "成交量Z分数", "rsi_7": "RSI(7)", "rsi_14": "RSI(14)",
    "atr_14": "ATR(14)", "bb_width": "布林带宽度",
    "price_position_14": "价格位置(14)", "close_vs_ma5": "偏离MA5",
    "close_vs_ma13": "偏离MA13", "close_vs_ma7": "偏离MA7",
    "close_vs_ma21": "偏离MA21", "close_vs_ema12": "偏离EMA12",
    "close_vs_ema26": "偏离EMA26", "trend_strength": "趋势强度",
}


def resolve_feature_names(model, df_columns):
    """Extract the actual feature names used by the trained model (after preprocessing)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor") or model.named_steps.get("columntransformer")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out(df_columns))

    return df_columns


def translate_names(names):
    result = []
    for name in names:
        for eng, cn in FEATURE_CN_MAP.items():
            if eng in name:
                base = name.replace(eng, cn).replace("num__", "").replace("cat__", "")
                result.append(base)
                break
        else:
            result.append(name.replace("num__", "").replace("cat__", ""))
    return result


def save_fig(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [√] {path.name}")
    return path


def main():
    model_path = ROOT_DIR.parent / "outputs" / "models" / "model_BTC_USDT_4h_20260527_125712.joblib"
    feature_csv = ROOT_DIR.parent / "outputs" / "ml_features_BTC_USDT_4h_20260527_125712.csv"
    out_dir = ROOT_DIR.parent / "outputs" / "shap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("特征重要性 & SHAP 高级分析")
    print("=" * 60)
    print(f"模型: {model_path.name}")
    print(f"特征数据: {feature_csv.name}")

    print("\n[1/6] 加载模型和数据...")
    artifact = load_trained_model(str(model_path))
    model: Pipeline = artifact["model"]
    best_threshold = artifact.get("best_threshold", 0.5)
    print(f"  模型类型: {type(model.named_steps.get('model', model)).__name__}")
    print(f"  最优阈值: {best_threshold}")

    df = pd.read_csv(feature_csv)
    raw_feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    config = MLConfig(symbol="BTC/USDT", timeframe="4h", target_horizon=6, target_threshold=0.0015)
    train_df, val_df, test_df = time_split(df, config.train_ratio, config.val_ratio)
    print(f"  训练集: {len(train_df)}  验证集: {len(val_df)}  测试集: {len(test_df)}")

    X_train = train_df[raw_feature_cols]
    X_test = test_df[raw_feature_cols]

    print("\n[2/6] 获取模型处理后特征名...")
    preprocessor = model.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out(raw_feature_cols))
    print(f"  处理后特征数: {len(feature_names)} (原始 {len(raw_feature_cols)} 列 → OneHot 展开 {len(feature_names)} 列)")

    X_train_t = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_t = pd.DataFrame(X_test_transformed, columns=feature_names)
    y_test = test_df["target_up"].values

    print("\n[3/6] 提取 XGBoost 内置特征重要性...")
    xgb_model = model.named_steps["model"]
    importances = xgb_model.feature_importances_
    top_n = min(20, len(feature_names))
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  {'排名':<5} {'特征名':<30} {'重要性':<10}")
    print(f"  {'-' * 45}")
    for rank, idx in enumerate(sorted_idx[:top_n], 1):
        print(f"  {rank:<5} {feature_names[idx]:<30} {importances[idx]:.6f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = translate_names([feature_names[i] for i in sorted_idx[:top_n][::-1]])
    top_importances = importances[sorted_idx[:top_n][::-1]]
    colors = plt.cm.Blues(0.4 + 0.6 * top_importances / top_importances.max())
    ax.barh(range(top_n), top_importances, color=colors, edgecolor="#1a5276", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("特征重要性 (Feature Importance)", fontsize=12)
    ax.set_title("XGBoost 内置特征重要性 Top-20\n(BTC/USDT 4h, horizon=6)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(top_importances):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    save_fig(fig, "01_xgboost_feature_importance.png", out_dir)

    print("\n[4/6] SHAP TreeExplainer 分析...")
    explainer = shap.TreeExplainer(
        xgb_model,
        feature_perturbation="interventional",
        data=X_train_t.values,
    )
    shap_values = explainer.shap_values(X_test_t.values)
    print(f"  SHAP values shape: {shap_values.shape}")

    print("\n[5/6] 生成 SHAP 图表...")

    cn_feature_names = translate_names(feature_names)

    fig = plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values, X_test_t.values, feature_names=cn_feature_names,
        max_display=20, show=False,
    )
    plt.title("SHAP Summary Plot — 特征对涨跌预测的贡献分布\n(BTC/USDT 4h, horizon=6)", fontsize=14, fontweight="bold")
    save_fig(fig, "02_shap_summary_beeswarm.png", out_dir)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test_t.values, feature_names=cn_feature_names,
        plot_type="bar", max_display=20, show=False,
    )
    ax.set_title("SHAP 特征重要性排序 (mean|SHAP|)\n(BTC/USDT 4h, horizon=6)", fontsize=14, fontweight="bold")
    save_fig(fig, "03_shap_bar_importance.png", out_dir)

    top_shap_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:8]
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    axes = axes.flatten()
    for i, feat_idx in enumerate(top_shap_idx):
        ax = axes[i]
        shap.dependence_plot(
            feat_idx, shap_values, X_test_t.values,
            feature_names=cn_feature_names,
            interaction_index=None, ax=ax, show=False,
        )
        ax.set_title(f"{cn_feature_names[feat_idx]}", fontsize=12, fontweight="bold")
        ax.set_xlabel(cn_feature_names[feat_idx], fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
    fig.suptitle("SHAP Dependence Plots — Top-8 特征与预测贡献的关系", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "04_shap_dependence_top8.png", out_dir)

    sample_idx = 0
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test_t.values[sample_idx],
            feature_names=cn_feature_names,
        ),
        show=False,
    )
    actual_label = y_test[sample_idx]
    prob = model.predict_proba(X_test.iloc[[sample_idx]])[0, 1]
    plt.title(
        f"SHAP Waterfall — 单样本预测解释 (样本#{sample_idx})\n"
        f"实际标签={'涨' if actual_label else '跌'} | 预测概率={prob:.3f}",
        fontsize=13, fontweight="bold",
    )
    save_fig(fig, "05_shap_waterfall_single.png", out_dir)

    print("\n[6/6] 特征分布对比分析 (涨 vs 跌)...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()
    top_features_eng = [feature_names[i] for i in top_shap_idx[:9]]
    y_train = train_df["target_up"].values

    for i, feat in enumerate(top_features_eng):
        ax = axes[i]
        idx = feature_names.index(feat)
        up_vals = X_test_t.values[y_test == 1, idx]
        down_vals = X_test_t.values[y_test == 0, idx]
        bp = ax.boxplot(
            [down_vals, up_vals],
            patch_artist=True,
            widths=0.5,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["跌 (0)", "涨 (1)"])
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#2ecc71")
        bp["boxes"][1].set_alpha(0.6)
        cn_name = cn_feature_names[idx]
        up_mean = np.mean(up_vals)
        down_mean = np.mean(down_vals)
        diff = up_mean - down_mean
        ax.set_title(f"{cn_name}\n(涨均值={up_mean:.3f}, 跌均值={down_mean:.3f}, Δ={diff:+.3f})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("特征分布对比：上涨 vs 下跌 样本（测试集）", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06_feature_distribution_up_vs_down.png", out_dir)

    top3_shap_features = [feature_names[i] for i in top_shap_idx[:3]]
    top3_cn = [cn_feature_names[i] for i in top_shap_idx[:3]]

    lines = ["## 附录：SHAP 特征重要性分析详细数据", "", ""]
    lines.append("### A. XGBoost 内置特征重要性 Top-20")
    lines.append("")
    lines.append("| 排名 | 特征名 | 重要性 (Gain) |")
    lines.append("|------|--------|---------------|")
    for rank, idx in enumerate(sorted_idx[:top_n], 1):
        lines.append(f"| {rank} | {cn_feature_names[idx]} | {importances[idx]:.6f} |")

    lines.append("")
    lines.append("### B. SHAP mean(|SHAP|) Top-20")
    lines.append("")
    lines.append("| 排名 | 特征名 | mean(|SHAP|) |")
    lines.append("|------|--------|-------------|")
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_sorted_idx = np.argsort(mean_shap)[::-1]
    for rank, idx in enumerate(shap_sorted_idx[:top_n], 1):
        lines.append(f"| {rank} | {cn_feature_names[idx]} | {mean_shap[idx]:.6f} |")

    lines.append("")
    lines.append(f"### C. Top-3 特征分布统计 (测试集)")
    lines.append("")
    for feat, cn_name in zip(top3_shap_features, top3_cn):
        idx = feature_names.index(feat)
        up_vals = X_test_t.values[y_test == 1, idx]
        down_vals = X_test_t.values[y_test == 0, idx]
        lines.append(f"**{cn_name}**")
        lines.append(f"")
        lines.append(f"| 统计量 | 下跌样本 (n={len(down_vals)}) | 上涨样本 (n={len(up_vals)}) | 差异 |")
        lines.append(f"|--------|----------------------------|----------------------------|------|")
        lines.append(f"| 均值 | {np.mean(down_vals):.4f} | {np.mean(up_vals):.4f} | {np.mean(up_vals)-np.mean(down_vals):+.4f} |")
        lines.append(f"| 中位数 | {np.median(down_vals):.4f} | {np.median(up_vals):.4f} | {np.median(up_vals)-np.median(down_vals):+.4f} |")
        lines.append(f"| 标准差 | {np.std(down_vals):.4f} | {np.std(up_vals):.4f} | {np.std(up_vals)-np.std(down_vals):+.4f} |")

    data_path = out_dir / "shap_analysis_data.txt"
    data_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [√] 分析数据保存至 {data_path}")

    print("\n" + "=" * 60)
    print("分析完成！所有图表保存至:", out_dir)
    print(f"共生成 6 张图表 + 1 份数据文件")
    return out_dir, feature_names, cn_feature_names, importances, shap_values, mean_shap


if __name__ == "__main__":
    main()