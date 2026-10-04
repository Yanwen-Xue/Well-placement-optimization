from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


CODE_ROOT = Path(r"F:\CODE1")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from validation_tools.train_component_f3_surrogate import (  # noqa: E402
    _fit_models,
)
from validation_tools.train_surrogate import build_feature_extractor  # noqa: E402


ROOT = CODE_ROOT / "_organized_active_work_20260817"
OUT = ROOT / "07_80year_three_target_analysis_0_683"
RESULTS = OUT / "results_0_683_three_targets_80y.npz"
EVOLUTION = OUT / "three_targets_30y_80y_679.csv"
GEOLOGY = ROOT / (
    r"06_verified_sandbody_analysis_0_683"
    r"\layout_petrophysical_connectivity_metrics_679_success.csv"
)
SEEDS = CODE_ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz"
PARAM_REPORT = ROOT / (
    r"02_current_333dim_proxy_0_683"
    r"\component_f3_metrics_0_683_static_well_triview_log_20260817.json"
)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rho = spearmanr(actual, predicted).statistic
    ape = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-12)
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "spearman_rho": float(rho),
    }


def predict_components(models, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = np.exp(np.asarray(models["pressure_std_mpa"].predict(features), dtype=float))
    mean = np.exp(np.asarray(models["pressure_mean_mpa"].predict(features), dtype=float))
    return std, mean


def layout_geometry(raw_x: np.ndarray) -> pd.DataFrame:
    rows = []
    for values in raw_x:
        injectors = values[:10].reshape(5, 2) * 50.0
        producers = values[10:].reshape(5, 2) * 50.0

        def group_metrics(points: np.ndarray) -> tuple[float, float]:
            distances = [
                np.linalg.norm(points[i] - points[j])
                for i in range(5)
                for j in range(i + 1, 5)
            ]
            return float(np.mean(distances)), float(ConvexHull(points).volume / 1.0e6)

        inj_pair, inj_hull = group_metrics(injectors)
        prod_pair, prod_hull = group_metrics(producers)
        source_sink = np.linalg.norm(
            injectors[:, None, :] - producers[None, :, :], axis=2
        )
        rows.append(
            {
                "inj_pair_mean_m": inj_pair,
                "inj_hull_km2": inj_hull,
                "prod_pair_mean_m": prod_pair,
                "prod_hull_km2": prod_hull,
                "inj_prod_nearest_mean_m": float(source_sink.min(axis=1).mean()),
                "centroid_distance_m": float(
                    np.linalg.norm(injectors.mean(axis=0) - producers.mean(axis=0))
                ),
            }
        )
    return pd.DataFrame(rows)


def set_style() -> None:
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def main() -> None:
    data = np.load(RESULTS, allow_pickle=True)
    finite = np.asarray(data["success_mask"], dtype=bool)
    raw_x = np.asarray(data["X"][finite], dtype=float)
    targets = np.asarray(data["objectives"][finite], dtype=float)
    pressure_std = np.asarray(
        data["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float
    )
    pressure_mean = np.asarray(
        data["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float
    )
    sample_index = np.asarray(data["sample_index"][finite], dtype=int)

    extractor, feature_report = build_feature_extractor(
        SEEDS, "prescreen_geometry_local_static_well_triview"
    )
    features = np.hstack([raw_x, np.asarray(extractor(raw_x), dtype=float)])
    params = json.loads(PARAM_REPORT.read_text(encoding="utf-8"))["xgb_params"]
    params["n_jobs"] = 16

    component_prediction = np.full_like(targets, np.nan)
    direct_log_f3_prediction = np.full(len(targets), np.nan)
    std_prediction = np.full(len(targets), np.nan)
    mean_prediction = np.full(len(targets), np.nan)
    fold_number = np.full(len(targets), -1, dtype=int)
    fold_rows = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features), start=1):
        models = _fit_models(
            features[train_idx],
            targets[train_idx],
            pressure_std[train_idx],
            pressure_mean[train_idx],
            params,
            log_components=True,
        )
        pred_std, pred_mean = predict_components(models, features[test_idx])
        pred = np.column_stack(
            [
                models["F1"].predict(features[test_idx]),
                models["F2"].predict(features[test_idx]),
                pred_std / pred_mean,
            ]
        )
        direct_model = XGBRegressor(**params)
        direct_model.fit(features[train_idx], np.log(targets[train_idx, 2]))
        direct_pred = np.exp(direct_model.predict(features[test_idx]))

        component_prediction[test_idx] = pred
        direct_log_f3_prediction[test_idx] = direct_pred
        std_prediction[test_idx] = pred_std
        mean_prediction[test_idx] = pred_mean
        fold_number[test_idx] = fold
        for target_index, target_name in enumerate(["F1_80", "F2_80", "F3_80_component"]):
            fold_rows.append(
                {
                    "fold": fold,
                    "target": target_name,
                    **metrics(targets[test_idx, target_index], pred[:, target_index]),
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "target": "F3_80_direct_log",
                **metrics(targets[test_idx, 2], direct_pred),
            }
        )

    pooled = {
        "F1_80": metrics(targets[:, 0], component_prediction[:, 0]),
        "F2_80": metrics(targets[:, 1], component_prediction[:, 1]),
        "F3_80_component": metrics(targets[:, 2], component_prediction[:, 2]),
        "F3_80_direct_log": metrics(targets[:, 2], direct_log_f3_prediction),
        "pressure_std_80": metrics(pressure_std, std_prediction),
        "pressure_mean_80": metrics(pressure_mean, mean_prediction),
    }
    top_two = np.argsort(targets[:, 2])[-2:]
    sensitivity_mask = np.ones(len(targets), dtype=bool)
    sensitivity_mask[top_two] = False
    pooled["F3_80_component_without_two_largest"] = metrics(
        targets[sensitivity_mask, 2], component_prediction[sensitivity_mask, 2]
    )

    oof = pd.DataFrame({"sample_index": sample_index, "fold": fold_number})
    for idx, name in enumerate(["F1_80", "F2_80", "F3_80"]):
        oof[f"actual_{name}"] = targets[:, idx]
        oof[f"oof_component_{name}"] = component_prediction[:, idx]
    oof["oof_direct_log_F3_80"] = direct_log_f3_prediction
    oof["actual_pressure_std_80_mpa"] = pressure_std
    oof["oof_pressure_std_80_mpa"] = std_prediction
    oof["actual_pressure_mean_80_mpa"] = pressure_mean
    oof["oof_pressure_mean_80_mpa"] = mean_prediction
    oof.to_csv(OUT / "surrogate_oof_predictions_80y_333dim.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "surrogate_oof_metrics_by_fold_80y.csv", index=False)

    evolution = pd.read_csv(EVOLUTION)
    geology = pd.read_csv(GEOLOGY)
    combined = evolution.merge(geology, on="sample_index", suffixes=("", "_geo"))
    geometry = layout_geometry(raw_x)
    geometry.insert(0, "sample_index", sample_index)
    combined = combined.merge(geometry, on="sample_index")
    fit = np.polyfit(combined["F2_30y_km2"], combined["F2_80y_km2"], 1)
    combined["F2_80_residual_from_30"] = combined["F2_80y_km2"] - np.polyval(
        fit, combined["F2_30y_km2"]
    )

    descriptors = [
        "inj_column_net_thickness_m_mean",
        "inj_column_bar_thickness_fraction_mean",
        "inj_column_kh_mD_m_mean",
        "inj_column_phih_m_mean",
        "inj_column_thickness_weighted_porosity_mean",
        "inj_column_longest_lowk_run_m_mean",
        "inj_column_lowk_run_count_mean",
        "prod_column_net_thickness_m_mean",
        "prod_column_bar_thickness_fraction_mean",
        "prod_column_kh_mD_m_mean",
        "prod_column_phih_m_mean",
        "prod_column_thickness_weighted_porosity_mean",
        "prod_column_longest_lowk_run_m_mean",
        "prod_column_lowk_run_count_mean",
        "inj_pair_mean_m",
        "inj_hull_km2",
        "prod_pair_mean_m",
        "prod_hull_km2",
        "inj_prod_nearest_mean_m",
        "centroid_distance_m",
        "total_brine_production",
    ]
    response_names = [
        "F2_80y_km2",
        "F2_change_pct",
        "F2_80_residual_from_30",
        "pressure_mean_80y_mpa",
        "pressure_std_80y_mpa",
        "F3_80y",
        "pressure_mean_change_pct",
        "pressure_std_change_pct",
    ]
    correlations = []
    for response in response_names:
        for descriptor in descriptors:
            rho = spearmanr(combined[descriptor], combined[response], nan_policy="omit").statistic
            correlations.append(
                {"response": response, "descriptor": descriptor, "spearman_rho": rho}
            )
    pd.DataFrame(correlations).to_csv(
        OUT / "geological_correlations_80y_selected_descriptors.csv", index=False
    )

    rank_summary = {}
    for metric in ["F2", "F3"]:
        before = combined[f"{metric}_30y" + ("_km2" if metric == "F2" else "")]
        after = combined[f"{metric}_80y" + ("_km2" if metric == "F2" else "")]
        count = int(np.floor(0.10 * len(combined)))
        low_before = set(combined.loc[before.nsmallest(count).index, "sample_index"])
        low_after = set(combined.loc[after.nsmallest(count).index, "sample_index"])
        high_before = set(combined.loc[before.nlargest(count).index, "sample_index"])
        high_after = set(combined.loc[after.nlargest(count).index, "sample_index"])
        rank_summary[metric] = {
            "spearman_30_to_80": float(spearmanr(before, after).statistic),
            "lowest_decile_overlap": float(len(low_before & low_after) / count),
            "highest_decile_overlap": float(len(high_before & high_after) / count),
        }

    decomposition = {}
    for year in [30, 80]:
        log_std = np.log(combined[f"pressure_std_{year}y_mpa"])
        log_mean = np.log(combined[f"pressure_mean_{year}y_mpa"])
        log_ratio = log_std - log_mean
        decomposition[str(year)] = {
            "variance_log_std": float(log_std.var(ddof=1)),
            "variance_log_mean": float(log_mean.var(ddof=1)),
            "covariance_log_std_log_mean": float(np.cov(log_std, log_mean, ddof=1)[0, 1]),
            "variance_log_ratio": float(log_ratio.var(ddof=1)),
            "rho_ratio_std": float(spearmanr(np.exp(log_ratio), np.exp(log_std)).statistic),
            "rho_ratio_mean": float(spearmanr(np.exp(log_ratio), np.exp(log_mean)).statistic),
        }

    summary = {
        "n_samples": int(len(targets)),
        "n_features": int(features.shape[1]),
        "feature_mode": feature_report["mode"],
        "pooled_oof_metrics": pooled,
        "rank_stability": rank_summary,
        "log_ratio_decomposition": decomposition,
        "cross_metric_spearman": {
            "30y_F1_F2": float(spearmanr(combined.F1_peak_80y, combined.F2_30y_km2).statistic),
            "30y_F1_F3": float(spearmanr(combined.F1_peak_80y, combined.F3_30y).statistic),
            "30y_F2_F3": float(spearmanr(combined.F2_30y_km2, combined.F3_30y).statistic),
            "80y_F1_F2": float(spearmanr(combined.F1_peak_80y, combined.F2_80y_km2).statistic),
            "80y_F1_F3": float(spearmanr(combined.F1_peak_80y, combined.F3_80y).statistic),
            "80y_F2_F3": float(spearmanr(combined.F2_80y_km2, combined.F3_80y).statistic),
        },
    }
    (OUT / "surrogate_and_mechanism_summary_80y.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65))
    labels = ["Peak average pressure increase (-)", "80-year plume area (km$^2$)",
              "80-year normalized pressure heterogeneity (-)"]
    for idx, axis in enumerate(axes):
        actual = targets[:, idx]
        predicted = component_prediction[:, idx]
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        pad = 0.04 * (upper - lower)
        axis.scatter(actual, predicted, s=11, color="#3E7C8D", alpha=0.68,
                     edgecolors="white", linewidths=0.25)
        axis.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
                  color="#333333", linestyle="--", linewidth=0.9)
        axis.set_xlim(lower - pad, upper + pad)
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_xlabel(f"CMG {labels[idx]}")
        axis.set_ylabel(f"OOF prediction {labels[idx]}")
        axis.set_title(f"({chr(97 + idx)})", loc="left")
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
        score = pooled[["F1_80", "F2_80", "F3_80_component"][idx]]
        axis.text(0.04, 0.95, f"$R^2$ = {score['r2']:.3f}\n$\\rho$ = {score['spearman_rho']:.3f}",
                  transform=axis.transAxes, va="top", ha="left")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.23, top=0.95, wspace=0.38)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"surrogate_oof_validation_80y.{suffix}", dpi=600 if suffix == "png" else None)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))
    panels = [
        (combined.F2_30y_km2, combined.F2_80y_km2, "30-year plume area (km$^2$)",
         "80-year plume area (km$^2$)"),
        (combined.pressure_mean_30y_mpa, combined.pressure_mean_80y_mpa,
         "30-year pressure mean (MPa)", "80-year pressure mean (MPa)"),
        (combined.F3_30y, combined.F3_80y, "30-year normalized heterogeneity (-)",
         "80-year normalized heterogeneity (-)"),
    ]
    for idx, (x, y, xlabel, ylabel) in enumerate(panels):
        axes[idx].scatter(x, y, s=11, color=["#B48758", "#55738F", "#3E7C8D"][idx],
                          alpha=0.68, edgecolors="white", linewidths=0.25)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(f"({chr(97 + idx)}) $\\rho$ = {spearmanr(x, y).statistic:.3f}", loc="left")
        axes[idx].grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.23, top=0.95, wspace=0.38)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"response_evolution_30y_to_80y.{suffix}", dpi=600 if suffix == "png" else None)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
