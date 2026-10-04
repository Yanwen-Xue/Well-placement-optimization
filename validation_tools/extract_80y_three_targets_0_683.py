from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


NI, NJ = 268, 203
INJ_K0, INJ_K1 = 40, 79
TARGET_DAY = 80.0 * 365.25
CELL_AREA_KM2 = 50.0 * 50.0 / 1.0e6

ROOT = Path(r"F:\CODE1\_organized_active_work_20260817")
SOURCE = ROOT / (
    r"training_data_0_683_333dim"
    r"\results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
)
OUT_DIR = ROOT / "07_80year_three_target_analysis_0_683"
CSV_OUT = OUT_DIR / "three_targets_30y_80y_679.csv"
NPZ_OUT = OUT_DIR / "results_0_683_three_targets_80y.npz"
SUMMARY_OUT = OUT_DIR / "three_targets_80y_extraction_summary.json"
PARTIAL_OUT = OUT_DIR / "three_targets_80y_partial.csv"


def nearest_spatial_group(handle: h5py.File, target_day: float) -> tuple[str, float]:
    days = handle["General/MasterTimeTable"][()]["Offset in days"].astype(float)
    groups = [name for name in handle["SpatialProperties"].keys() if name.isdigit()]
    group = min(groups, key=lambda name: abs(days[int(name)] - target_day))
    return group, float(days[int(group)])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True)
    success = np.asarray(data["success_mask"], dtype=bool)
    sample_index = np.asarray(data["sample_index"], dtype=int)
    successful_rows = np.where(success)[0]

    reference_row = int(successful_rows[0])
    reference_sr3 = Path(str(data["run_dir"][reference_row])) / "opt_001.sr3"
    with h5py.File(reference_sr3, "r") as handle:
        grid = handle["SpatialProperties/000000/GRID"]
        ipstcs = grid["IPSTCS"][()].astype(np.int64)
        block_pv = grid["BLOCKPVOL"][()].astype(np.float64)
        full0 = ipstcs - 1
        layer = full0 // (NI * NJ)
        active = (
            (layer >= INJ_K0)
            & (layer <= INJ_K1)
            & np.isfinite(block_pv)
            & (block_pv > 0.0)
        )
        interval_idx = np.where(active)[0]
        p0_mpa = (
            handle["SpatialProperties/000000/PRES"][()][interval_idx].astype(np.float64)
            / 1000.0
        )

    interval_pv = block_pv[interval_idx]
    full_interval = full0[interval_idx]
    column_id = (full_interval % NI) + NI * ((full_interval // NI) % NJ)

    rows: list[dict] = []
    errors: list[dict] = []
    started = time.time()
    for count, row_index in enumerate(successful_rows, start=1):
        case_id = int(sample_index[row_index])
        sr3_path = Path(str(data["run_dir"][row_index])) / "opt_001.sr3"
        try:
            with h5py.File(sr3_path, "r") as handle:
                group, selected_day = nearest_spatial_group(handle, TARGET_DAY)
                root = handle[f"SpatialProperties/{group}"]
                sg = root["SG"][()][interval_idx].astype(np.float64)
                pressure_mpa = (
                    root["PRES"][()][interval_idx].astype(np.float64) / 1000.0
                )

            plume = np.isfinite(sg) & (sg >= 0.1)
            plume_columns = int(np.unique(column_id[plume]).size)
            plume_area = float(plume_columns * CELL_AREA_KM2)

            pressure_change = pressure_mpa - p0_mpa
            finite_pressure = (
                np.isfinite(pressure_change)
                & np.isfinite(interval_pv)
                & (interval_pv > 0.0)
            )
            weights = interval_pv[finite_pressure]
            values = pressure_change[finite_pressure]
            weight_sum = float(weights.sum())
            pressure_mean = float(np.sum(weights * values) / weight_sum)
            pressure_variance = float(
                np.sum(weights * (values - pressure_mean) ** 2) / weight_sum
            )
            pressure_std = float(np.sqrt(max(pressure_variance, 0.0)))
            pressure_cv = (
                float(pressure_std / pressure_mean)
                if pressure_mean > 0.0
                else float("nan")
            )

            f1 = float(data["objectives"][row_index, 0])
            f2_30 = float(data["plume_area_sg0p1_km2"][row_index])
            mean_30 = float(data["pressure_heterogeneity_injend_mean_mpa"][row_index])
            std_30 = float(data["pressure_heterogeneity_injend_std_mpa"][row_index])
            f3_30 = float(std_30 / mean_30)
            rows.append(
                {
                    "sample_index": case_id,
                    "sr3_path": str(sr3_path),
                    "time_day_80": selected_day,
                    "F1_peak_80y": f1,
                    "F2_30y_km2": f2_30,
                    "F2_80y_km2": plume_area,
                    "F2_change_km2": plume_area - f2_30,
                    "F2_change_pct": 100.0 * (plume_area / f2_30 - 1.0),
                    "pressure_mean_30y_mpa": mean_30,
                    "pressure_mean_80y_mpa": pressure_mean,
                    "pressure_mean_change_pct": 100.0 * (pressure_mean / mean_30 - 1.0),
                    "pressure_std_30y_mpa": std_30,
                    "pressure_std_80y_mpa": pressure_std,
                    "pressure_std_change_pct": 100.0 * (pressure_std / std_30 - 1.0),
                    "F3_30y": f3_30,
                    "F3_80y": pressure_cv,
                    "F3_change_pct": 100.0 * (pressure_cv / f3_30 - 1.0),
                    "plume_columns_80y": plume_columns,
                }
            )
        except Exception as exc:
            errors.append(
                {"sample_index": case_id, "sr3_path": str(sr3_path), "error": repr(exc)}
            )

        if count % 25 == 0 or count == len(successful_rows):
            pd.DataFrame(rows).sort_values("sample_index").to_csv(PARTIAL_OUT, index=False)
            elapsed = time.time() - started
            rate = count / elapsed if elapsed > 0.0 else float("nan")
            print(
                f"processed={count}/{len(successful_rows)} "
                f"ok={len(rows)} errors={len(errors)} rate={rate:.2f} cases/s",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    frame.to_csv(CSV_OUT, index=False)

    f2_80 = np.full(len(success), np.nan, dtype=float)
    mean_80 = np.full(len(success), np.nan, dtype=float)
    std_80 = np.full(len(success), np.nan, dtype=float)
    f3_80 = np.full(len(success), np.nan, dtype=float)
    time_80 = np.full(len(success), np.nan, dtype=float)
    sample_to_row = {int(value): idx for idx, value in enumerate(sample_index)}
    for record in frame.itertuples(index=False):
        idx = sample_to_row[int(record.sample_index)]
        f2_80[idx] = float(record.F2_80y_km2)
        mean_80[idx] = float(record.pressure_mean_80y_mpa)
        std_80[idx] = float(record.pressure_std_80y_mpa)
        f3_80[idx] = float(record.F3_80y)
        time_80[idx] = float(record.time_day_80)

    objectives_80 = np.column_stack(
        [np.asarray(data["objectives"][:, 0], dtype=float), f2_80, f3_80]
    )
    finite_80 = success & np.isfinite(objectives_80).all(axis=1) & (mean_80 > 0.0)
    np.savez_compressed(
        NPZ_OUT,
        sample_index=sample_index,
        X=np.asarray(data["X"], dtype=float),
        objectives=objectives_80,
        success_mask=finite_80,
        original_success_mask=success,
        pressure_heterogeneity_injend_std_mpa=std_80,
        pressure_heterogeneity_injend_mean_mpa=mean_80,
        plume_area_sg0p1_km2=f2_80,
        plume_day=time_80,
        run_dir=np.asarray(data["run_dir"]),
    )

    summary = {
        "source": str(SOURCE),
        "target_year": 80,
        "target_day_nominal": TARGET_DAY,
        "n_original_success": int(success.sum()),
        "n_extracted": int(len(frame)),
        "n_finite_three_target_rows": int(finite_80.sum()),
        "n_errors": int(len(errors)),
        "errors": errors,
        "runtime_seconds": float(time.time() - started),
        "definitions": {
            "F1": "maximum reservoir-average relative pressure increase over 80 years",
            "F2": "80-year horizontal projection of K41-K80 cells with SG >= 0.1",
            "F3": "80-year PV-weighted std(deltaP) / mean(deltaP) over K41-K80",
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if PARTIAL_OUT.exists():
        PARTIAL_OUT.unlink()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
