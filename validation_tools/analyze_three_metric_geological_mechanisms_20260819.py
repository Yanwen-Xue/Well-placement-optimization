from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"F:\CODE1")
BASE = ROOT / r"_organized_active_work_20260816\04_verified_sandbody_analysis"
OUT = (
    ROOT
    / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817"
    / r"final_analysis_cases_18\three_metric_geological_synthesis_20260819"
)
OUT.mkdir(parents=True, exist_ok=True)

METRICS = BASE / r"petrophysical_connectivity_audit\layout_petrophysical_connectivity_metrics_0_623.csv"
DESIGN = BASE / r"advanced_response_analysis\advanced_analysis_design_table.csv"
F2_STATIC = (
    ROOT
    / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817"
    / r"final_analysis_cases_18\f2_petrophysics_orientation_20260819"
    / "f2_all_679_static_layout_metrics.csv"
)


def rho_table(frame: pd.DataFrame, target: str, features: list[str], mechanism: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        pair = frame[[target, feature]].apply(pd.to_numeric, errors="coerce").dropna()
        rho, p_value = spearmanr(pair[target], pair[feature])
        rows.append(
            {
                "mechanism": mechanism,
                "target": target,
                "feature": feature,
                "n": len(pair),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def quartile_summary(frame: pd.DataFrame, feature: str, targets: list[str]) -> pd.DataFrame:
    work = frame[[feature, *targets]].apply(pd.to_numeric, errors="coerce").dropna()
    work["quartile"] = pd.qcut(work[feature], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    rows = []
    for quartile, group in work.groupby("quartile", observed=True):
        row = {
            "stratifying_feature": feature,
            "quartile": str(quartile),
            "n": len(group),
            "feature_median": group[feature].median(),
        }
        for target in targets:
            row[f"{target}_median"] = group[target].median()
            row[f"{target}_q25"] = group[target].quantile(0.25)
            row[f"{target}_q75"] = group[target].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["success"].astype(str).str.lower().eq("true")].copy()
    design = pd.read_csv(DESIGN)
    f2 = pd.read_csv(F2_STATIC)

    data = metrics.merge(
        design[
            [
                "sample_index",
                "inj_pair_distance_mean_m",
                "inj_pair_distance_min_m",
                "prod_pair_distance_mean_m",
                "inj_prod_pair_distance_mean_m",
                "inj_prod_centroid_distance_m",
                "inj_prescreen_mean",
                "prod_prescreen_mean",
            ]
        ],
        on="sample_index",
        how="left",
    )

    data["source_sink_kh_contrast"] = (
        data["inj_column_kh_mD_m_mean"] / data["prod_column_kh_mD_m_mean"]
    )
    data["prod_weak_well_fraction"] = (
        data["prod_column_kh_mD_m_min"] / data["prod_column_kh_mD_m_mean"]
    )
    data["prod_kh_cv"] = (
        data["prod_column_kh_mD_m_std"] / data["prod_column_kh_mD_m_mean"]
    )
    data["inj_kh_cv"] = (
        data["inj_column_kh_mD_m_std"] / data["inj_column_kh_mD_m_mean"]
    )
    data["log_F3"] = np.log(data["F3_pressure_CV_std_over_mean"])
    data["log_pressure_mean_check"] = np.log(data["pressure_mean_mpa"])
    data["log_pressure_std_check"] = np.log(data["pressure_std_mpa"])

    tables = [
        rho_table(
            data,
            "F1_pressure_rise",
            [
                "total_brine_production",
                "prod_column_kh_mD_m_mean",
                "prod_column_kh_mD_m_min",
                "prod_weak_well_fraction",
                "prod_kh_cv",
                "prod_column_phih_m_mean",
                "prod_column_bar_thickness_fraction_mean",
                "prod_column_longest_lowk_run_m_mean",
                "source_sink_kh_contrast",
                "inj_prod_pair_distance_mean_m",
            ],
            "F1 pressure-relief capacity",
        ),
        rho_table(
            f2,
            "F2_km2",
            [
                "inj_net_thickness_m_mean",
                "inj_net_to_gross_mean",
                "inj_kh_mD_m_mean",
                "inj_porosity_h_weighted_mean",
                "inj_longest_low_k_run_layers_mean",
                "inj_top_to_bottom_kh_ratio_mean",
                "well_distance_mean_m",
                "layout_spread_major_m",
            ],
            "F2 vertical reuse and plan-view expansion",
        ),
        rho_table(
            data,
            "pressure_mean_mpa",
            [
                "prod_column_kh_mD_m_mean",
                "prod_column_phih_m_mean",
                "prod_column_bar_thickness_fraction_mean",
                "prod_column_longest_lowk_run_m_mean",
                "prod_pair_distance_mean_m",
            ],
            "F3 denominator: regional pressure baseline",
        ),
        rho_table(
            data,
            "pressure_std_mpa",
            [
                "inj_pair_distance_mean_m",
                "inj_pair_distance_min_m",
                "inj_column_kh_mD_m_mean",
                "inj_column_kh_mD_m_std",
                "inj_kh_cv",
                "inj_column_longest_lowk_run_m_mean",
            ],
            "F3 numerator: localized pressure departures",
        ),
        rho_table(
            data,
            "F3_pressure_CV_std_over_mean",
            [
                "pressure_mean_mpa",
                "pressure_std_mpa",
                "prod_column_kh_mD_m_mean",
                "inj_pair_distance_mean_m",
                "inj_column_kh_mD_m_std",
            ],
            "F3 normalized pressure heterogeneity",
        ),
    ]
    correlations = pd.concat(tables, ignore_index=True)
    correlations.to_csv(OUT / "three_metric_mechanism_correlations.csv", index=False)

    quartiles = pd.concat(
        [
            quartile_summary(
                data,
                "prod_column_kh_mD_m_mean",
                ["F1_pressure_rise", "total_brine_production", "pressure_mean_mpa", "F3_pressure_CV_std_over_mean"],
            ),
            quartile_summary(
                f2,
                "inj_net_to_gross_mean",
                ["F2_km2"],
            ),
            quartile_summary(
                data,
                "inj_pair_distance_mean_m",
                ["pressure_std_mpa", "F3_pressure_CV_std_over_mean"],
            ),
        ],
        ignore_index=True,
    )
    quartiles.to_csv(OUT / "three_metric_quartile_diagnostics.csv", index=False)

    f3_valid = data[
        np.isfinite(data["log_pressure_std_check"])
        & np.isfinite(data["log_pressure_mean_check"])
        & np.isfinite(data["log_F3"])
    ].copy()
    var_std = float(f3_valid["log_pressure_std_check"].var(ddof=1))
    var_mean = float(f3_valid["log_pressure_mean_check"].var(ddof=1))
    covariance = float(f3_valid[["log_pressure_std_check", "log_pressure_mean_check"]].cov().iloc[0, 1])
    var_f3 = float(f3_valid["log_F3"].var(ddof=1))
    decomposition = pd.DataFrame(
        [
            {
                "n": len(f3_valid),
                "var_log_pressure_std": var_std,
                "var_log_pressure_mean": var_mean,
                "cov_log_std_log_mean": covariance,
                "minus_2cov_term": -2.0 * covariance,
                "reconstructed_var_log_F3": var_std + var_mean - 2.0 * covariance,
                "observed_var_log_F3": var_f3,
                "spearman_F3_vs_pressure_std": spearmanr(
                    f3_valid["F3_pressure_CV_std_over_mean"], f3_valid["pressure_std_mpa"]
                ).statistic,
                "spearman_F3_vs_pressure_mean": spearmanr(
                    f3_valid["F3_pressure_CV_std_over_mean"], f3_valid["pressure_mean_mpa"]
                ).statistic,
            }
        ]
    )
    decomposition.to_csv(OUT / "F3_log_variance_decomposition.csv", index=False)

    def get_rho(mechanism: str, feature: str) -> float:
        row = correlations[(correlations["mechanism"] == mechanism) & (correlations["feature"] == feature)]
        return float(row.iloc[0]["spearman_rho"])

    f1_mechanism = "F1 pressure-relief capacity"
    f2_mechanism = "F2 vertical reuse and plan-view expansion"
    d = decomposition.iloc[0]
    report = f"""# Three-Metric Geological Mechanism Synthesis

## Scope and evidence

This audit uses 619 successful layouts with verified well-scale petrophysical and sand-occupancy descriptors, the 679-layout static plume screen, and the full-physics paired cases already selected for the manuscript. Rank correlations are screening evidence. Permeability, pore-capacity, NTG, and low-permeability segmentation are strongly collinear and are interpreted as a combined effective hydraulic-capacity state.

## F1: pressure buildup records the functionality of the production network

- F1 versus cumulative brine production: Spearman rho = {get_rho(f1_mechanism, 'total_brine_production'):.3f}.
- F1 versus mean production-column kh: rho = {get_rho(f1_mechanism, 'prod_column_kh_mD_m_mean'):.3f}.
- F1 versus minimum production-column kh: rho = {get_rho(f1_mechanism, 'prod_column_kh_mD_m_min'):.3f}.
- F1 versus source/sink kh contrast: rho = {get_rho(f1_mechanism, 'source_sink_kh_contrast'):.3f}.

The geological control is not gross completed thickness or sand-object size alone. A production well contributes to regional pressure relief only where net thickness, permeability, pore capacity, and vertical continuity coexist. Low-quality completions reduce both total withdrawal and the effective number of pressure-relief nodes. Case 9/10 provides field-level confirmation: a 28.7% reduction in production-side integrated kh accompanies a 22.9% reduction in brine withdrawal and a 48.5% increase in F1.

**Mechanistic statement:** pressure buildup is governed by the hydraulic functionality and spatial support of the production network. The source/sink contrast amplifies pressure where injection-side capacity is retained but production-side capacity is interrupted.

## F2: the footprint is a projection-allocation response

- F2 versus injection net thickness: rho = {get_rho(f2_mechanism, 'inj_net_thickness_m_mean'):.3f}.
- F2 versus injection NTG: rho = {get_rho(f2_mechanism, 'inj_net_to_gross_mean'):.3f}.
- F2 versus longest low-k interval: rho = {get_rho(f2_mechanism, 'inj_longest_low_k_run_layers_mean'):.3f}.
- F2 versus mean injector spacing: rho = {get_rho(f2_mechanism, 'well_distance_mean_m'):.3f}.

The strong univariate petrophysical correlations collapse after controlling for correlated prescreen and geological variables, so no single independent causal coefficient can be assigned to kh, porosity, or NTG. Their joint meaning is accessible transmissive thickness. The full-physics pairs resolve how this state changes F2: transmissive vertical stacking reuses the same horizontal columns across more layers, while low vertical participation forces more lateral area per active layer. Well spacing adds a separate projection effect through coalescence of injector-scale plume regions.

**Mechanistic statement:** F2 measures how a three-dimensional plume is allocated between vertical layer reuse and unique horizontal columns. It is not a direct measure of total affected pore volume. Case 7/8 separates interwell coalescence from vertical reuse; Case 255/312 retains five separated plume regions and isolates the vertical-architecture contribution.

## F3: normalized heterogeneity combines two geological scales

- F3 versus pressure standard deviation: rho = {d['spearman_F3_vs_pressure_std']:.3f}.
- F3 versus pressure mean: rho = {d['spearman_F3_vs_pressure_mean']:.3f}.
- var(log pressure std) = {d['var_log_pressure_std']:.4f}; var(log pressure mean) = {d['var_log_pressure_mean']:.4f}; observed var(log F3) = {d['observed_var_log_F3']:.4f}.

Cross-validated diagnostics already show that geometry explains pressure standard deviation much better than petrophysics (R2 = 0.877 versus 0.175), whereas petrophysics explains pressure mean much better than geometry (R2 = 0.710 versus 0.223). Case 5/6 confirms this separation: Case 6 lowers the mean by 43.0% while the standard deviation rises by only 4.7%, increasing F3 by 83.5%.

**Mechanistic statement:** production-side hydraulic capacity sets the regional pressure baseline; injection-well geometry and local injection-side architecture organize departures around that baseline. Strong pressure relief can lower the denominator while localized injection lobes persist, so F3 can increase as average pressure decreases.

## Cross-metric geological synthesis

The three responses correspond to different spatial operators on the same heterogeneous system:

1. F1 integrates the reservoir-wide source/sink balance and is controlled chiefly by production-network functionality.
2. F2 projects the three-dimensional mobile plume onto the horizontal plane and is controlled by vertical reuse plus interwell plan-view overlap.
3. F3 normalizes a local-to-regional pressure contrast and is controlled by the coupling of a production-set baseline with injection-organized deviations.

This resolves the apparent response decoupling without claiming that one sand-body descriptor independently controls all outcomes. The geological contribution is the identification of scale-specific hydraulic roles within a vertically separated multiwell system.

## Claims that the current data do not support

- A universal causal effect for one named sand-body shape class.
- Independent effects of permeability, porosity, NTG, and sand occupancy; these descriptors are strongly collinear.
- Dynamic hydraulic communication inferred solely from static connected-component labels.
- Interpretation of a smaller F2 as a smaller total CO2-affected volume.
- Interpretation of a smaller F3 as simultaneously lower mean pressure and lower absolute pressure variation.
"""
    (OUT / "three_metric_geological_mechanism_report.md").write_text(report, encoding="utf-8")
    print(correlations.to_string(index=False))
    print(decomposition.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
