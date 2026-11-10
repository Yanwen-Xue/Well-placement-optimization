from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import spearmanr


ROOT = Path(r"F:\CODE1")
RESULTS = ROOT / r"_organized_active_work_20260817\training_data_0_683_333dim\results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
CASE_ROOT = ROOT / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817\final_analysis_cases_18"
NSGA_ROOT = CASE_ROOT.parent
OUT = CASE_ROOT / "f2_petrophysics_orientation_20260819"
OUT.mkdir(parents=True, exist_ok=True)

NI, NJ, NK = 268, 203, 200
K0, K1 = 40, 80
TARGET_DAY = 30.0 * 365.25
CELL_AREA_KM2 = 0.0025

sys.path.insert(0, str(ROOT))
import cmg_runner  # noqa: E402


CASE_PATHS = {
    7: CASE_ROOT / r"machine_5_final_analysis_07\lhs_doe\case_0007_F2_pair_F2_low\run_001\opt_001.sr3",
    8: CASE_ROOT / r"machine_5_final_analysis_07\lhs_doe\case_0008_F2_pair_F2_high\run_001\opt_001.sr3",
    255: CASE_ROOT / r"machine_5_final_analysis_07\lhs_doe\case_0255\run_001\opt_001.sr3",
    312: CASE_ROOT / r"machine_5_final_analysis_07\lhs_doe\case_0312\run_001\opt_001.sr3",
}


def load_property(path: Path) -> np.ndarray:
    return cmg_runner._read_property_txt(str(path), NI, NJ, NK)


def parse_dat_wells(dat_path: Path) -> np.ndarray:
    text = dat_path.read_text(encoding="utf-8", errors="replace")
    wells = []
    for number in range(1, 6):
        match = re.search(
            rf"PERF\s+GEOA\s+'INJ-{number}'\s*\n\*\*[^\n]*\n\s*(\d+)\s+(\d+)\s+\d+",
            text,
        )
        if match is None:
            raise ValueError(f"Cannot locate INJ-{number} in {dat_path}")
        wells.append((int(match.group(1)) - 1, int(match.group(2)) - 1))
    return np.asarray(wells, dtype=int)


def selected_case_wells() -> dict[int, np.ndarray]:
    manifest = json.loads((CASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for case in (7, 8):
        item = next(row for row in manifest["cases"] if int(row["case_id"]) == case)
        result[case] = np.asarray(item["inj_wells_0based"], dtype=int)
    for case in (255, 312):
        result[case] = parse_dat_wells(CASE_PATHS[case].with_suffix(".dat"))
    return result


def count_runs(mask: np.ndarray) -> tuple[int, int]:
    runs = 0
    longest = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
        else:
            if current:
                runs += 1
                longest = max(longest, current)
            current = 0
    if current:
        runs += 1
        longest = max(longest, current)
    return runs, longest


def well_static_metrics(
    wells: np.ndarray,
    perm: np.ndarray,
    poro: np.ndarray,
    thick: np.ndarray,
    sand_kij: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for number, (i, j) in enumerate(wells, 1):
        h = np.asarray(thick[i, j, K0:K1], dtype=float)
        k = np.asarray(perm[i, j, K0:K1], dtype=float)
        phi = np.asarray(poro[i, j, K0:K1], dtype=float)
        sand = np.asarray(sand_kij[K0:K1, i, j], dtype=bool)
        valid = np.isfinite(h) & (h > 0) & np.isfinite(k) & (k > 0) & np.isfinite(phi)
        gross = float(np.sum(h[valid]))
        net = float(np.sum(h[valid & sand]))
        kh = float(np.sum(k[valid] * h[valid]))
        phih = float(np.sum(phi[valid] * h[valid]))
        high_k = valid & (k >= 4.0)
        high_runs, longest_high_run = count_runs(high_k)
        low_runs, longest_low_run = count_runs(valid & (k < 4.0))
        midpoint = (K1 - K0) // 2
        top_kh = float(np.sum(k[:midpoint][valid[:midpoint]] * h[:midpoint][valid[:midpoint]]))
        bottom_kh = float(np.sum(k[midpoint:][valid[midpoint:]] * h[midpoint:][valid[midpoint:]]))
        rows.append(
            {
                "well": f"I{number}",
                "i_0based": int(i),
                "j_0based": int(j),
                "net_to_gross": net / gross if gross else np.nan,
                "net_thickness_m": net,
                "kh_mD_m": kh,
                "kh_mean_mD": kh / gross if gross else np.nan,
                "porosity_h_weighted": phih / gross if gross else np.nan,
                "high_k_layer_fraction": float(np.mean(high_k)),
                "high_k_run_count": high_runs,
                "longest_high_k_run_layers": longest_high_run,
                "low_k_run_count": low_runs,
                "longest_low_k_run_layers": longest_low_run,
                "top_to_bottom_kh_ratio": top_kh / bottom_kh if bottom_kh > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def layout_geometry(wells: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> dict[str, float]:
    xy = np.asarray([[cx[i, j], cy[i, j]] for i, j in wells], dtype=float)
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    tri = distances[np.triu_indices(5, 1)]
    centered = xy - xy.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T @ centered / len(xy))
    return {
        "well_distance_mean_m": float(np.mean(tri)),
        "well_distance_min_m": float(np.min(tri)),
        "well_distance_cv": float(np.std(tri) / np.mean(tri)),
        "layout_spread_major_m": float(np.sqrt(max(eig[-1], 0.0))),
        "layout_spread_minor_m": float(np.sqrt(max(eig[0], 0.0))),
    }


def plume_occupancy(sr3_path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(sr3_path, "r") as handle:
        days = np.asarray(handle["General/MasterTimeTable"]["Offset in days"], dtype=float)
        groups = sorted(
            (name for name in handle["SpatialProperties"].keys() if name.isdigit()),
            key=int,
        )
        group_ids = np.asarray([int(name) for name in groups], dtype=int)
        group = groups[int(np.argmin(np.abs(days[group_ids] - TARGET_DAY)))]
        grid = handle["SpatialProperties/000000/GRID"]
        ipstcs = np.asarray(grid["IPSTCS"], dtype=np.int64) - 1
        pv = np.asarray(grid["BLOCKPVOL"], dtype=float)
        full_i = ipstcs % NI
        full_j = (ipstcs // NI) % NJ
        full_k = ipstcs // (NI * NJ)
        sg = np.asarray(handle[f"SpatialProperties/{group}/SG"], dtype=float)
        valid = (
            (full_k >= K0)
            & (full_k < K1)
            & np.isfinite(pv)
            & (pv > 0)
            & np.isfinite(sg)
        )
        occupancy = np.zeros((K1 - K0, NI, NJ), dtype=bool)
        pick = valid & (sg >= 0.10)
        occupancy[full_k[pick] - K0, full_i[pick], full_j[pick]] = True
        return occupancy, float(days[int(group)])


def plume_shape_metrics(occupancy: np.ndarray) -> dict[str, float]:
    projection = np.any(occupancy, axis=0)
    layer_counts = np.sum(occupancy, axis=0)
    layer_areas = np.sum(occupancy, axis=(1, 2)) * CELL_AREA_KM2
    f2 = float(np.sum(projection) * CELL_AREA_KM2)
    a_sum = float(np.sum(layer_areas))
    occupied_columns = layer_counts[projection]
    layer_index = np.arange(K0 + 1, K1 + 1, dtype=float)
    if a_sum > 0:
        vertical_centroid = float(np.sum(layer_index * layer_areas) / a_sum)
        vertical_std = float(np.sqrt(np.sum((layer_index - vertical_centroid) ** 2 * layer_areas) / a_sum))
    else:
        vertical_centroid = np.nan
        vertical_std = np.nan
    labels, component_count = label(projection, structure=np.ones((3, 3), dtype=np.uint8))
    component_areas = [float(np.sum(labels == n) * CELL_AREA_KM2) for n in range(1, component_count + 1)]
    return {
        "F2_km2": f2,
        "layer_area_sum_km2": a_sum,
        "mean_occupied_layers_per_column": a_sum / f2 if f2 else np.nan,
        "horizontal_fraction_F2_over_layer_sum": f2 / a_sum if a_sum else np.nan,
        "occupied_layer_count": int(np.count_nonzero(layer_areas)),
        "vertical_layer_centroid": vertical_centroid,
        "vertical_layer_std": vertical_std,
        "upper_half_plume_fraction": float(np.sum(layer_areas[:20]) / a_sum) if a_sum else np.nan,
        "column_layer_count_p10": float(np.percentile(occupied_columns, 10)) if occupied_columns.size else np.nan,
        "column_layer_count_median": float(np.median(occupied_columns)) if occupied_columns.size else np.nan,
        "column_layer_count_p90": float(np.percentile(occupied_columns, 90)) if occupied_columns.size else np.nan,
        "projected_component_count": int(component_count),
        "largest_component_fraction": max(component_areas) / f2 if component_areas and f2 else np.nan,
    }


def component_metrics(occupancy: np.ndarray, wells: np.ndarray) -> pd.DataFrame:
    projection = np.any(occupancy, axis=0)
    labels, component_count = label(projection, structure=np.ones((3, 3), dtype=np.uint8))
    well_labels = []
    occupied = np.argwhere(projection)
    for well in wells:
        if labels[tuple(well)] > 0:
            well_labels.append(int(labels[tuple(well)]))
        else:
            nearest = occupied[int(np.argmin(np.sum((occupied - well) ** 2, axis=1)))]
            well_labels.append(int(labels[tuple(nearest)]))
    rows = []
    for component in range(1, component_count + 1):
        component_projection = labels == component
        component_occupancy = occupancy & component_projection[None, :, :]
        metrics = plume_shape_metrics(component_occupancy)
        linked = [f"I{n + 1}" for n, lab in enumerate(well_labels) if lab == component]
        coords = np.argwhere(component_projection)
        centered = coords - coords.mean(axis=0)
        eig = np.linalg.eigvalsh(centered.T @ centered / len(coords)) if len(coords) > 1 else np.zeros(2)
        metrics.update(
            {
                "component": component,
                "linked_injection_wells": ";".join(linked),
                "horizontal_axis_ratio": float(np.sqrt(eig[-1] / eig[0])) if eig[0] > 0 else np.nan,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def all_sample_static_table(
    result: dict[str, np.ndarray],
    perm: np.ndarray,
    poro: np.ndarray,
    thick: np.ndarray,
    sand: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for pos, sample_index in enumerate(result["sample_index"]):
        if not bool(result["success_mask"][pos]) or not np.all(np.isfinite(result["objectives"][pos])):
            continue
        wells = np.asarray(result["inj_wells"][pos], dtype=int)
        static = well_static_metrics(wells, perm, poro, thick, sand)
        geometry = layout_geometry(wells, cx, cy)
        row = {
            "sample_index": int(sample_index),
            "F2_km2": float(result["objectives"][pos, 1]),
            **geometry,
        }
        for column in (
            "net_to_gross",
            "net_thickness_m",
            "kh_mD_m",
            "kh_mean_mD",
            "porosity_h_weighted",
            "high_k_layer_fraction",
            "high_k_run_count",
            "longest_high_k_run_layers",
            "low_k_run_count",
            "longest_low_k_run_layers",
            "top_to_bottom_kh_ratio",
        ):
            row[f"inj_{column}_mean"] = float(static[column].mean())
            row[f"inj_{column}_std"] = float(static[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in samples.columns:
        if column in {"sample_index", "F2_km2"}:
            continue
        valid = np.isfinite(samples[column]) & np.isfinite(samples["F2_km2"])
        rho, pvalue = spearmanr(samples.loc[valid, column], samples.loc[valid, "F2_km2"])
        rows.append({"feature": column, "n": int(valid.sum()), "spearman_rho": rho, "p_value": pvalue})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda values: np.abs(values), ascending=False)


def pareto_static_table(
    perm: np.ndarray,
    poro: np.ndarray,
    thick: np.ndarray,
    sand: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> pd.DataFrame:
    solutions = json.loads((NSGA_ROOT / "full_pareto_solutions.json").read_text(encoding="utf-8"))
    rows = []
    for solution in solutions:
        wells = np.asarray(solution["inj_wells"], dtype=int)
        static = well_static_metrics(wells, perm, poro, thick, sand)
        row = {
            "pareto_rank": int(solution["rank"]),
            "pred_F1": float(solution["F1_pressure_rise"]),
            "pred_F2_km2": float(solution["F2_plume_area_sg0p1_km2"]),
            "pred_F3": float(solution["F3_pressure_CV_std_over_mean"]),
            **layout_geometry(wells, cx, cy),
        }
        for column in (
            "net_to_gross",
            "net_thickness_m",
            "kh_mD_m",
            "kh_mean_mD",
            "porosity_h_weighted",
            "high_k_layer_fraction",
            "longest_high_k_run_layers",
            "longest_low_k_run_layers",
            "top_to_bottom_kh_ratio",
        ):
            row[f"inj_{column}_mean"] = float(static[column].mean())
            row[f"inj_{column}_std"] = float(static[column].std(ddof=0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    # These scores are screening labels only. Dynamic plume orientation requires SR3 output.
    frame["static_horizontal_tendency_score"] = (
        frame["inj_longest_low_k_run_layers_mean"].rank(pct=True)
        + (1.0 - frame["inj_net_to_gross_mean"].rank(pct=True))
        + (1.0 - frame["inj_kh_mean_mD_mean"].rank(pct=True))
        + frame["well_distance_mean_m"].rank(pct=True)
    ) / 4.0
    frame["static_vertical_reuse_tendency_score"] = (
        frame["inj_net_to_gross_mean"].rank(pct=True)
        + frame["inj_kh_mean_mD_mean"].rank(pct=True)
        + frame["inj_longest_high_k_run_layers_mean"].rank(pct=True)
        + (1.0 - frame["inj_longest_low_k_run_layers_mean"].rank(pct=True))
    ) / 4.0
    return frame


def main() -> None:
    result = dict(np.load(RESULTS, allow_pickle=True))
    seeds = dict(np.load(ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz", allow_pickle=True))
    perm = load_property(ROOT / r"property_CMG\property_260421_all_perm.txt")
    poro = load_property(ROOT / r"property_CMG\property_260421_all_poro.txt")
    thick = np.asarray(seeds["thick_3d"], dtype=float)
    sand = np.asarray(
        np.load(
            ROOT / r"_organized_active_work_20260816\04_verified_sandbody_analysis\verified_sandbody_grid_268x203x200.npz",
            allow_pickle=True,
        )["sand_mask_kij"],
        dtype=bool,
    )
    cx = np.asarray(seeds["cx_3d"][:, :, K0], dtype=float)
    cy = np.asarray(seeds["cy_3d"][:, :, K0], dtype=float)

    samples = all_sample_static_table(result, perm, poro, thick, sand, cx, cy)
    correlations = correlation_table(samples)
    samples.to_csv(OUT / "f2_all_679_static_layout_metrics.csv", index=False)
    correlations.to_csv(OUT / "f2_static_geometry_spearman_correlations.csv", index=False)

    pareto = pareto_static_table(perm, poro, thick, sand, cx, cy)
    pareto.to_csv(OUT / "f2_full_pareto_static_petrophysics_screen.csv", index=False)
    pareto.nlargest(15, "static_horizontal_tendency_score").to_csv(
        OUT / "f2_pareto_horizontal_tendency_candidates.csv", index=False
    )
    pareto.nlargest(15, "static_vertical_reuse_tendency_score").to_csv(
        OUT / "f2_pareto_vertical_reuse_tendency_candidates.csv", index=False
    )

    wells_by_case = selected_case_wells()
    case_rows = []
    well_rows = []
    component_frames = []
    for case, path in CASE_PATHS.items():
        wells = wells_by_case[case]
        static = well_static_metrics(wells, perm, poro, thick, sand)
        static.insert(0, "case", case)
        occupancy, plume_day = plume_occupancy(path)
        shape = plume_shape_metrics(occupancy)
        case_rows.append(
            {
                "case": case,
                "plume_day": plume_day,
                **shape,
                **layout_geometry(wells, cx, cy),
                "inj_net_to_gross_mean": float(static.net_to_gross.mean()),
                "inj_kh_mean_mD_mean": float(static.kh_mean_mD.mean()),
                "inj_porosity_h_weighted_mean": float(static.porosity_h_weighted.mean()),
                "inj_longest_low_k_run_layers_mean": float(static.longest_low_k_run_layers.mean()),
                "inj_top_to_bottom_kh_ratio_mean": float(static.top_to_bottom_kh_ratio.mean()),
            }
        )
        components = component_metrics(occupancy, wells)
        components.insert(0, "case", case)
        component_frames.append(components)
        label_by_well = {}
        for _, component in components.iterrows():
            for well in str(component.linked_injection_wells).split(";"):
                if well:
                    label_by_well[well] = component.to_dict()
        for _, row in static.iterrows():
            dynamic = label_by_well.get(row.well, {})
            well_rows.append({**row.to_dict(), **{f"plume_{key}": value for key, value in dynamic.items()}})

    cases = pd.DataFrame(case_rows)
    wells = pd.DataFrame(well_rows)
    components = pd.concat(component_frames, ignore_index=True)
    cases.to_csv(OUT / "f2_selected_case_orientation_summary.csv", index=False)
    wells.to_csv(OUT / "f2_selected_well_static_and_plume_metrics.csv", index=False)
    components.to_csv(OUT / "f2_selected_projected_component_metrics.csv", index=False)

    selected = wells[((wells.case == 255) & (wells.well == "I4")) | ((wells.case == 312) & (wells.well == "I5"))]
    selected.to_csv(OUT / "f2_case255_I4_case312_I5_detailed_comparison.csv", index=False)

    print("Top absolute Spearman associations with F2:")
    print(correlations.head(15).to_string(index=False))
    print("\nSelected case orientation summary:")
    print(cases.to_string(index=False))
    print("\nCase 255-I4 and Case 312-I5:")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
