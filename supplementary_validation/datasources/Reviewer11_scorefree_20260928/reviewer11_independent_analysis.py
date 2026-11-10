"""Score-free geological diagnosis against the 679 GEM layout responses."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from jcp_public_repository.validation_tools.build_static_sideview_fields import read_cmg_txt


ROOT = Path(__file__).resolve().parent
SOURCE = Path("F:/CODE1/_organized_active_work_20260817/06_verified_sandbody_analysis_0_683")
OUT = ROOT / "reviewer11_new_analysis"


def geometry(x):
    inj = x[:10].reshape(5, 2) * 50.0
    prod = x[10:20].reshape(5, 2) * 50.0
    def pairs(wells):
        return np.mean([np.linalg.norm(wells[i] - wells[j]) for i in range(5) for j in range(i + 1, 5)])
    cross = np.linalg.norm(inj[:, None, :] - prod[None, :, :], axis=2)
    return pairs(inj), pairs(prod), np.mean(np.min(cross, axis=1)), np.linalg.norm(inj.mean(0) - prod.mean(0))


def partial_rank(x, y, controls):
    ranked = np.column_stack([np.ones(len(x)), *(rankdata(c) for c in controls.T)])
    rx = rankdata(x) - ranked @ np.linalg.lstsq(ranked, rankdata(x), rcond=None)[0]
    ry = rankdata(y) - ranked @ np.linalg.lstsq(ranked, rankdata(y), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_partial(x, y, controls, seed=42, repeats=1000):
    rng = np.random.default_rng(seed)
    n = len(x)
    estimates = [partial_rank(x[idx], y[idx], controls[idx]) for idx in (rng.integers(n, size=n) for _ in range(repeats))]
    return np.percentile(estimates, [2.5, 97.5]).tolist()


def longest_injection_sand(x, sand, thickness):
    values = []
    for i, j in np.rint(x[:10]).astype(int).reshape(5, 2):
        run = longest = 0.0
        for is_sand, h in zip(sand[40:80, i, j], thickness[i, j, 40:80]):
            run = run + h if is_sand else 0.0
            longest = max(longest, run)
        values.append(longest)
    return float(np.mean(values))


def main():
    OUT.mkdir(exist_ok=True)
    sand = pd.read_csv(SOURCE / "verified_well_layout_sandbody_features_679_success.csv")
    pet = pd.read_csv(SOURCE / "layout_petrophysical_connectivity_metrics_679_success.csv")
    with np.load(ROOT / "jcp_public_repository/public_data/training_responses_684.npz") as z:
        source = pd.DataFrame({"sample_index": z["sample_index"], "success": z["success_mask"]})
        positions = dict(zip(z["sample_index"].tolist(), z["X"]))
        outcomes = dict(zip(z["sample_index"].tolist(), z["objectives"]))
    d = pet.merge(sand[["sample_index", "inj_local_sand_fraction_500m_mean", "prod_local_sand_fraction_500m_mean"]], on="sample_index", validate="one_to_one")
    assert len(d) == 679 and d.sample_index.is_unique
    assert d.sample_index.isin(source.loc[source.success, "sample_index"]).all()
    assert np.allclose(d[["F1_pressure_rise", "F2_plume_area_sg0p1_km2", "F3_pressure_CV_std_over_mean"]],
                       np.array([outcomes[i] for i in d.sample_index]), rtol=1e-7, atol=1e-9)
    assert np.allclose(d.pressure_std_mpa / d.pressure_mean_mpa, d.F3_pressure_CV_std_over_mean)
    d[["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "source_sink_centroid_m"]] = np.array([geometry(positions[i]) for i in d.sample_index])
    with np.load(SOURCE / "verified_sandbody_grid_268x203x200.npz") as z:
        sand_mask = np.asarray(z["sand_mask_kij"], dtype=bool)
    thickness = read_cmg_txt(Path("F:/CODE1/property_CMG/2026_0421_1day_property Grid Thickness 2025-Jan-01.txt"))
    d["inj_longest_sand_m"] = [longest_injection_sand(positions[i], sand_mask, thickness) for i in d.sample_index]

    specs = [
        ("F1", "prod_column_kh_mD_m_mean", "F1_pressure_rise", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m"]),
        ("F1_local", "prod_local_sand_fraction_500m_mean", "F1_pressure_rise", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m"]),
        ("F1_connectivity", "prod_kge6_well_pairs_sharing_component", "F1_pressure_rise", ["prod_column_kh_mD_m_mean", "inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m"]),
        ("F2", "inj_column_longest_lowk_run_m_mean", "F2_plume_area_sg0p1_km2", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "inj_column_kh_mD_m_mean"]),
        ("F2_sand", "inj_longest_sand_m", "F2_plume_area_sg0p1_km2", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "inj_column_kh_mD_m_mean"]),
        ("F2_local", "inj_local_sand_fraction_500m_mean", "F2_plume_area_sg0p1_km2", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "inj_column_kh_mD_m_mean"]),
        ("F2_spacing", "inj_spacing_m", "F2_plume_area_sg0p1_km2", ["inj_column_kh_mD_m_mean", "inj_column_longest_lowk_run_m_mean", "prod_spacing_m"]),
        ("F3_mean", "prod_column_kh_mD_m_mean", "pressure_mean_mpa", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "inj_column_kh_mD_m_mean"]),
        ("F3_std", "inj_local_sand_fraction_500m_mean", "pressure_std_mpa", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "prod_column_kh_mD_m_mean", "inj_column_kh_mD_m_mean"]),
        ("F3_offset_std", "source_sink_centroid_m", "pressure_std_mpa", ["inj_spacing_m", "prod_spacing_m", "prod_column_kh_mD_m_mean", "inj_column_kh_mD_m_mean", "inj_local_sand_fraction_500m_mean"]),
        ("F3_offset_mean", "source_sink_centroid_m", "pressure_mean_mpa", ["inj_spacing_m", "prod_spacing_m", "prod_column_kh_mD_m_mean", "inj_column_kh_mD_m_mean", "inj_local_sand_fraction_500m_mean"]),
        ("F3_ratio_prod", "prod_column_kh_mD_m_mean", "F3_pressure_CV_std_over_mean", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "inj_column_kh_mD_m_mean"]),
        ("F3_offset_ratio", "source_sink_centroid_m", "F3_pressure_CV_std_over_mean", ["inj_spacing_m", "prod_spacing_m", "prod_column_kh_mD_m_mean", "inj_column_kh_mD_m_mean", "inj_local_sand_fraction_500m_mean"]),
        ("F3_ratio_inj", "inj_local_sand_fraction_500m_mean", "F3_pressure_CV_std_over_mean", ["inj_spacing_m", "prod_spacing_m", "inj_nearest_prod_m", "prod_column_kh_mD_m_mean", "inj_column_kh_mD_m_mean"]),
    ]
    records = []
    for label, feature, target, control_names in specs:
        x, y = d[feature].to_numpy(), d[target].to_numpy()
        controls = d[control_names].to_numpy()
        ci_low, ci_high = boot_partial(x, y, controls)
        records.append({"analysis": label, "feature": feature, "response": target,
                        "controls": ", ".join(control_names), "n": len(d),
                        "spearman_rho": float(spearmanr(x, y).statistic),
                        "partial_rank_r": partial_rank(x, y, controls),
                        "bootstrap_95_low": ci_low,
                        "bootstrap_95_high": ci_high})
    pd.DataFrame(records).to_csv(OUT / "gem_scorefree_associations.csv", index=False)

    quartiles = []
    for feature, targets in (
        ("prod_column_kh_mD_m_mean", ["F1_pressure_rise", "pressure_mean_mpa", "pressure_std_mpa", "F3_pressure_CV_std_over_mean"]),
        ("inj_column_longest_lowk_run_m_mean", ["F2_plume_area_sg0p1_km2"]),
        ("inj_longest_sand_m", ["F2_plume_area_sg0p1_km2"]),
        ("inj_local_sand_fraction_500m_mean", ["F2_plume_area_sg0p1_km2", "pressure_std_mpa"]),
        ("inj_spacing_m", ["F2_plume_area_sg0p1_km2", "pressure_std_mpa"]),
        ("source_sink_centroid_m", ["pressure_mean_mpa", "pressure_std_mpa", "F3_pressure_CV_std_over_mean"]),
    ):
        bins = pd.qcut(d[feature], 4, labels=False, duplicates="drop")
        for target in targets:
            for q in sorted(bins.unique()):
                selected = d.loc[bins == q]
                quartiles.append({"feature": feature, "response": target, "quartile": int(q) + 1,
                                  "n": len(selected), "feature_median": selected[feature].median(),
                                  "response_median": selected[target].median(),
                                  "response_q25": selected[target].quantile(.25),
                                  "response_q75": selected[target].quantile(.75)})
    pd.DataFrame(quartiles).to_csv(OUT / "gem_scorefree_quartiles.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, feature, target, xlabel, ylabel in (
        (axes[0], "prod_column_kh_mD_m_mean", "F1_pressure_rise", "Production integrated kh (mD m)", "Peak pressure increase, F1"),
        (axes[1], "inj_longest_sand_m", "F2_plume_area_sg0p1_km2", "Injection continuous sand (m)", "Mobile CO2 footprint, F2 (km2)"),
        (axes[2], "prod_column_kh_mD_m_mean", "F3_pressure_CV_std_over_mean", "Production integrated kh (mD m)", "Pressure variability, F3"),
    ):
        ax.scatter(d[feature], d[target], s=8, alpha=.22, color="#287a77", rasterized=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "gem_scorefree_three_responses.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for ax, feature, target, color, xlabel, label in (
        (axes[0], "prod_column_kh_mD_m_mean", "pressure_mean_mpa", "#287a77", "Production integrated kh (mD m)", "Mean pressure increment (MPa)"),
        (axes[1], "source_sink_centroid_m", "pressure_std_mpa", "#b85f40", "Injection-production centroid offset (m)", "Pressure-increment SD (MPa)"),
    ):
        ax.scatter(d[feature], d[target], s=8, alpha=.24, color=color, rasterized=True)
        ax.set(xlabel=xlabel, ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "gem_f3_components.png", dpi=220)
    plt.close(fig)

    summary = {"n": len(d), "source": str(SOURCE), "response_check": "matches public training NPZ",
               "pair_component_counts_k6": d.prod_kge6_well_pairs_sharing_component.value_counts().sort_index().to_dict(),
               "pressure_ratio_check_max_abs": float(np.max(np.abs(d.pressure_std_mpa / d.pressure_mean_mpa - d.F3_pressure_CV_std_over_mean)))}
    (OUT / "gem_scorefree_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
