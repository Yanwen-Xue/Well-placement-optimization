from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


CODE_ROOT = Path(r"F:\CODE1")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from validation_tools.train_component_f3_surrogate import (  # noqa: E402
    _fit_models,
    _predict_from_models,
)
from validation_tools.train_surrogate import build_feature_extractor  # noqa: E402


RESULTS = CODE_ROOT / (
    r"_organized_active_work_20260817\training_data_0_683_333dim"
    r"\results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
)
SEEDS = CODE_ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz"
REPORT_DIR = CODE_ROOT / "_organized_active_work_20260817"
REPORTS = {
    "Coordinates only": REPORT_DIR
    / r"03_coord_only_ablation_0_683\component_f3_metrics_0_683_coords_log_20260817.json",
    "+ Plan-view prescreen": REPORT_DIR
    / r"04_prescreen_only_ablation_0_683\component_f3_metrics_0_683_prescreen_log_20260817.json",
    "Full geological input": REPORT_DIR
    / r"02_current_333dim_proxy_0_683\component_f3_metrics_0_683_static_well_triview_log_20260817.json",
}
OUT_DIR = Path(r"C:\Users\admin\Desktop\paper_figure\writen\figure")

TARGET_KEYS = ["F1", "F2", "F3"]
TARGET_LABELS = ["F₁", "F₂", "F₃"]
AXIS_LABELS = [
    "F₁ (%)",
    "F₂ (km²)",
    "F₃ (pressure heterogeneity)",
]
COLORS = ["#B9CAD6", "#82A6BC", "#4F7893"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman_rho": float(rho),
    }


def load_training_arrays():
    data = np.load(RESULTS, allow_pickle=True)
    success = np.asarray(data["success_mask"], dtype=bool)
    finite = (
        success
        & np.isfinite(data["objectives"]).all(axis=1)
        & np.isfinite(data["pressure_heterogeneity_injend_std_mpa"])
        & np.isfinite(data["pressure_heterogeneity_injend_mean_mpa"])
        & (data["pressure_heterogeneity_injend_mean_mpa"] > 0.0)
    )
    return (
        data,
        finite,
        np.asarray(data["X"][finite], dtype=float),
        np.asarray(data["objectives"][finite], dtype=float),
        np.asarray(data["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float),
        np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float),
    )


def make_oof_predictions():
    data, finite, raw_x, y, pressure_std, pressure_mean = load_training_arrays()
    report = json.loads(REPORTS["Full geological input"].read_text(encoding="utf-8"))
    extractor, feature_report = build_feature_extractor(
        SEEDS, "prescreen_geometry_local_static_well_triview"
    )
    x_features = np.hstack([raw_x, np.asarray(extractor(raw_x), dtype=float)])
    if x_features.shape != (679, 333):
        raise ValueError(f"Expected 679 x 333 features, got {x_features.shape}")

    params = dict(report["xgb_params"])
    params["n_jobs"] = 16
    predicted = np.full_like(y, np.nan, dtype=float)
    folds = np.full(len(y), -1, dtype=int)
    fold_rows = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_features), start=1):
        models = _fit_models(
            x_features[train_idx],
            y[train_idx],
            pressure_std[train_idx],
            pressure_mean[train_idx],
            params,
            log_components=True,
        )
        fold_pred = _predict_from_models(
            models, x_features[test_idx], log_components=True
        )
        predicted[test_idx] = fold_pred
        folds[test_idx] = fold
        for target_index, target in enumerate(TARGET_KEYS):
            row = {"fold": fold, "target": target}
            row.update(metrics(y[test_idx, target_index], fold_pred[:, target_index]))
            fold_rows.append(row)

    if not np.isfinite(predicted).all() or (folds < 1).any():
        raise RuntimeError("OOF prediction matrix is incomplete")

    sample_index = np.asarray(data["sample_index"][finite], dtype=int)
    frame = pd.DataFrame({"sample_index": sample_index, "fold": folds})
    pooled_rows = []
    for target_index, target in enumerate(TARGET_KEYS):
        frame[f"actual_{target}"] = y[:, target_index]
        frame[f"oof_pred_{target}"] = predicted[:, target_index]
        frame[f"residual_{target}"] = predicted[:, target_index] - y[:, target_index]
        row = {"target": target, "n": len(y)}
        row.update(metrics(y[:, target_index], predicted[:, target_index]))
        pooled_rows.append(row)

    frame.to_csv(OUT_DIR / "surrogate_oof_predictions_0_683.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUT_DIR / "surrogate_oof_metrics_by_fold_0_683.csv", index=False
    )
    pd.DataFrame(pooled_rows).to_csv(
        OUT_DIR / "surrogate_oof_metrics_pooled_0_683.csv", index=False
    )
    metadata = {
        "n_samples": int(len(y)),
        "n_features": int(x_features.shape[1]),
        "folds": 5,
        "shuffle": True,
        "random_state": 42,
        "f3_training": "log(std) and log(mean), reconstructed as std/mean",
        "feature_mode": feature_report["mode"],
        "pooled_metrics": pooled_rows,
    }
    (OUT_DIR / "surrogate_oof_validation_0_683.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return y, predicted, pooled_rows


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_oof_validation(y, predicted, pooled_rows):
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65))
    letters = ["(a)", "(b)", "(c)"]
    decimals = [3, 3, 3]
    for index, ax in enumerate(axes):
        actual = y[:, index]
        pred = predicted[:, index]
        pooled = pooled_rows[index]
        display_scale = 100.0 if index == 0 else 1.0
        actual = actual * display_scale
        pred = pred * display_scale
        lower = min(float(actual.min()), float(pred.min()))
        upper = max(float(actual.max()), float(pred.max()))
        padding = 0.045 * (upper - lower)
        limits = (lower - padding, upper + padding)
        ax.scatter(
            actual,
            pred,
            s=12,
            color="#3E7C8D",
            alpha=0.68,
            edgecolors="white",
            linewidths=0.25,
        )
        ax.plot(limits, limits, color="#202020", linewidth=1.0, linestyle="--")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"CMG {AXIS_LABELS[index]}")
        ax.set_ylabel(f"OOF prediction {AXIS_LABELS[index]}")
        ax.set_title(f"{letters[index]} {TARGET_LABELS[index]}", loc="left", fontweight="bold")
        rmse_text = f"{pooled['rmse'] * display_scale:.{decimals[index]}f}"
        mae_text = f"{pooled['mae'] * display_scale:.{decimals[index]}f}"
        ax.text(
            0.04,
            0.95,
            f"R² = {pooled['r2']:.3f}\nRMSE = {rmse_text}\nMAE = {mae_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#B7B7B7", "pad": 2.5},
        )
        ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.20, top=0.93, wspace=0.34)
    fig.savefig(OUT_DIR / "surrogate_oof_validation_0_683.png", dpi=600)
    fig.savefig(OUT_DIR / "surrogate_oof_validation_0_683.pdf")
    fig.savefig(OUT_DIR / "surrogate_oof_validation_0_683.svg")
    plt.close(fig)


def load_ablation_summary():
    rows = []
    for model_index, (label, path) in enumerate(REPORTS.items()):
        report = json.loads(path.read_text(encoding="utf-8"))
        extra = int(report["feature_report"].get("n_prescreen_features", 0))
        for target_index, target in enumerate(TARGET_KEYS):
            cv = report["cross_validation"][target]
            test = report["test_metrics"][target]
            rows.append(
                {
                    "model": label,
                    "model_index": model_index,
                    "feature_count": 20 + extra,
                    "target": target,
                    "target_index": target_index,
                    "cv_r2_mean": cv["r2"]["mean"],
                    "cv_r2_std": cv["r2"]["std"],
                    "cv_rmse_mean": cv["rmse"]["mean"],
                    "test_r2": test["r2"],
                    "test_rmse": test["rmse"],
                    "test_mae": test["mae"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "surrogate_feature_ablation_0_683.csv", index=False)
    return frame


def plot_ablation(frame: pd.DataFrame):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.85), sharey=True)
    x = np.arange(3, dtype=float)
    labels = ["20-D coordinates", "42-D + plan-view maps", "333-D full feature set"]
    models = list(REPORTS)
    width = 0.23
    offsets = (np.arange(len(models)) - 1) * width

    for model_index, (model, color) in enumerate(zip(models, COLORS)):
        subset = frame[frame["model"] == model].sort_values("target_index")
        cv_values = subset["cv_r2_mean"].to_numpy()
        cv_errors = subset["cv_r2_std"].to_numpy()
        holdout_values = subset["test_r2"].to_numpy()
        bar_x = x + offsets[model_index]
        axes[0].bar(
            bar_x,
            cv_values,
            width=width,
            yerr=cv_errors,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.75, "capsize": 2.0,
                      "capthick": 0.75},
            label=labels[model_index],
            zorder=3,
        )
        axes[1].bar(
            bar_x,
            holdout_values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        for target_index, (value, error) in enumerate(zip(cv_values, cv_errors)):
            axes[0].annotate(
                f"{value:.3f}",
                (bar_x[target_index], value + error),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#252525",
            )
        for target_index, value in enumerate(holdout_values):
            axes[1].annotate(
                f"{value:.3f}",
                (bar_x[target_index], value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#252525",
            )

    axes[0].set_title("(a) Five-fold cross-validation", loc="left", fontweight="normal", pad=8)
    axes[0].set_ylabel("R² (mean ± SD)")
    axes[1].set_title("(b) Holdout validation", loc="left", fontweight="normal", pad=8)
    for ax in axes:
        ax.set_xticks(x, TARGET_LABELS)
        ax.set_xlim(-0.52, 2.52)
        ax.set_ylim(0.0, 1.08)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.8, length=3.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.45,
        handlelength=1.7,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.78, wspace=0.16)
    fig.savefig(OUT_DIR / "surrogate_feature_ablation_0_683.png", dpi=600)
    fig.savefig(OUT_DIR / "surrogate_feature_ablation_0_683.pdf")
    fig.savefig(OUT_DIR / "surrogate_feature_ablation_0_683.svg")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y, predicted, pooled_rows = make_oof_predictions()
    plot_oof_validation(y, predicted, pooled_rows)
    ablation = load_ablation_summary()
    plot_ablation(ablation)
    print(json.dumps({"pooled_oof_metrics": pooled_rows}, indent=2))
    print(f"Saved figures and tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
