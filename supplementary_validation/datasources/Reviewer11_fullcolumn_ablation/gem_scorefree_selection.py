"""Summarize geological properties of layouts selected by GEM objectives alone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = Path("F:/CODE1")
SOURCE = CODE / "_organized_active_work_20260817/06_verified_sandbody_analysis_0_683"
RESULTS = CODE / "_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"


def geometry(x):
    inj = x[:10].reshape(5, 2) * 50.0
    prod = x[10:20].reshape(5, 2) * 50.0
    return float(np.linalg.norm(inj.mean(axis=0) - prod.mean(axis=0)))


def main():
    pet = pd.read_csv(SOURCE / "layout_petrophysical_connectivity_metrics_679_success.csv")
    sand = pd.read_csv(SOURCE / "verified_well_layout_sandbody_features_679_success.csv",
                       usecols=["sample_index", "inj_local_sand_fraction_500m_mean"])
    data = pet.merge(sand, on="sample_index", validate="one_to_one")
    with np.load(RESULTS, allow_pickle=False) as result:
        good = result["success_mask"].astype(bool)
        ids = result["sample_index"][good].astype(int)
        X = result["X"][good]
        y = result["objectives"][good]
    assert len(data) == len(ids) == 679
    assert data.sample_index.is_unique
    assert set(data.sample_index) == set(ids)
    lookup = dict(zip(ids, X))
    data["source_sink_centroid_m"] = [geometry(lookup[idx]) for idx in data.sample_index]
    order = {idx: row for row, idx in enumerate(ids)}
    assert np.allclose(data[["F1_pressure_rise", "F2_plume_area_sg0p1_km2",
                             "F3_pressure_CV_std_over_mean"]].to_numpy(),
                       y[[order[idx] for idx in data.sample_index]])
    assert np.allclose(data.pressure_std_mpa / data.pressure_mean_mpa,
                       data.F3_pressure_CV_std_over_mean)
    target_cols = {
        "F1": "F1_pressure_rise",
        "F2": "F2_plume_area_sg0p1_km2",
        "F3": "F3_pressure_CV_std_over_mean",
    }
    measures = {
        "F1": ["prod_column_kh_mD_m_mean", "prod_local_sand_fraction_500m_mean"],
        "F2": ["inj_local_sand_fraction_500m_mean", "inj_column_longest_lowk_run_m_mean",
               "inj_column_kh_mD_m_mean"],
        "F3": ["source_sink_centroid_m", "prod_column_kh_mD_m_mean",
               "pressure_std_mpa", "pressure_mean_mpa"],
    }
    # Production local sand fraction comes from the independent facies grid.
    if "prod_local_sand_fraction_500m_mean" not in data:
        more = pd.read_csv(SOURCE / "verified_well_layout_sandbody_features_679_success.csv",
                           usecols=["sample_index", "prod_local_sand_fraction_500m_mean"])
        data = data.merge(more, on="sample_index", validate="one_to_one")
    records = []
    selected = {}
    n_best = 136
    for target, target_col in target_cols.items():
        best = data.nsmallest(n_best, target_col)
        selected[target] = best.sample_index.astype(int).tolist()
        for measure in [target_col] + measures[target]:
            all_med = float(data[measure].median())
            best_med = float(best[measure].median())
            records.append({"target": target, "selection": f"lowest {n_best}/679 GEM {target} values",
                            "measure": measure, "all_679_median": all_med,
                            "gem_best_136_median": best_med,
                            "difference": best_med - all_med,
                            "relative_difference_pct": 100 * (best_med / all_med - 1) if all_med else None})
    pd.DataFrame(records).to_csv(HERE / "gem_scorefree_selection.csv", index=False)
    (HERE / "gem_scorefree_selection.json").write_text(
        json.dumps({"n_total": 679, "n_best_each": n_best,
                    "selection": "GEM response only; no prescreen scores or surrogate predictions",
                    "selected_sample_index": selected, "rows": records}, indent=2), encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
