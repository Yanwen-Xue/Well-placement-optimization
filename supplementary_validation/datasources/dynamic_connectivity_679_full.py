from pathlib import Path
import json
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(r"F:/CODE1")
RESULTS = ROOT / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
PEAKS = ROOT / r"_organized_active_work_20260817/07_80year_three_target_analysis_0_683/sr3_dynamic_pressure_extraction/sr3_pressure_peak_summary_679.csv"
OUTDIR = Path(r"C:/Users/admin/Desktop/paper_figure/writen/datasources/dynamic_connectivity_679_run")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "dynamic_connectivity_679.csv"
STATE = OUTDIR / "state.json"
LOG = OUTDIR / "run.log"

NI, NJ, NK = 268, 203, 200
PROD_K0, PROD_K1 = 0, 15
THRESHOLDS = (0.1, 0.5, 1.0)
STRUCTURE = ndimage.generate_binary_structure(3, 1)


def peak_group(f, peak_year):
    days = np.asarray(f["General/MasterTimeTable"]["Offset in days"], dtype=float)
    names = sorted((n for n in f["SpatialProperties"] if n.isdigit()), key=int)
    ids = np.asarray([int(n) for n in names], dtype=int)
    target = int(np.argmin(np.abs(days[ids] / 365.25 - float(peak_year))))
    return names[target], float(days[ids[target]] / 365.25)


def extract(path, peak_year, wells):
    wells = np.asarray(wells, dtype=int).reshape(5, 2)
    with h5py.File(path, "r") as f:
        group, used_year = peak_group(f, peak_year)
        grid = f["SpatialProperties/000000/GRID"]
        ip = np.asarray(grid["IPSTCS"], dtype=np.int64) - 1
        pv = np.asarray(grid["BLOCKPVOL"], dtype=np.float64)
        ii = ip % NI
        jj = (ip // NI) % NJ
        kk = ip // (NI * NJ)
        # The pressure-interference diagnostic is evaluated in the completed
        # producer interval. This isolates the pressure field that can be
        # directly accessed by the producer network from the injection uplift.
        active = np.isfinite(pv) & (pv > 0) & (kk >= PROD_K0) & (kk < PROD_K1)
        ids = np.flatnonzero(active)
        pt = np.asarray(f[f"SpatialProperties/{group}/PRES"][ids], dtype=np.float64)
        vol = pv[ids]
        i = ii[ids].astype(np.int32)
        j = jj[ids].astype(np.int32)
        k = kk[ids].astype(np.int32)
        finite = np.isfinite(pt) & np.isfinite(vol)
        pressure = pt[finite] / 1000.0
        vol = vol[finite]
        i, j, k = i[finite], j[finite], k[finite]
        # Dense 3-D masks retain the actual active-cell adjacency used for connectivity.
        flat = np.ravel_multi_index((k, j, i), (NK, NJ, NI))
        seed_mask = np.zeros(NK * NJ * NI, dtype=bool)
        seed_pressure = []
        for wi, wj in wells:
            sk = np.arange(PROD_K0, PROD_K1, dtype=np.int32)
            valid_k = sk[(wi >= 0) & (wi < NI) & (wj >= 0) & (wj < NJ)]
            seed_flat = np.ravel_multi_index((valid_k, np.full_like(valid_k, wj), np.full_like(valid_k, wi)), (NK, NJ, NI))
            seed_mask[seed_flat] = True
            hit = (i == wi) & (j == wj)
            seed_pressure.append(float(np.nanmean(pressure[hit])) if np.any(hit) else np.nan)
        seed_pressure = np.asarray(seed_pressure, dtype=float)
        xy_dist = np.sqrt((i[:, None] - wells[None, :, 0]) ** 2 + (j[:, None] - wells[None, :, 1]) ** 2)
        nearest = np.argmin(xy_dist, axis=1)
        # Contrast to the nearest producer pressure is a dynamic interference
        # measure and does not use the final objective value itself.
        pressure_contrast = pressure - seed_pressure[nearest]
        out = {"peak_group_year": used_year, "n_active_cells": int(len(pressure))}
        for threshold in THRESHOLDS:
            mask = pressure_contrast >= threshold
            mask_flat = np.zeros(NK * NJ * NI, dtype=bool)
            mask_flat[flat[mask]] = True
            seed_hit = seed_mask & mask_flat
            if not seed_hit.any():
                selected = np.zeros_like(mask_flat)
            else:
                labels, _ = ndimage.label(mask_flat.reshape((NK, NJ, NI)), structure=STRUCTURE)
                seed_labels = np.unique(labels.reshape(-1)[seed_hit])
                seed_labels = seed_labels[seed_labels > 0]
                selected = np.isin(labels.reshape(-1), seed_labels)
            chosen = selected[flat]
            key = str(threshold).replace(".", "p")
            out[f"connected_volume_{key}mpa_m3"] = float(vol[chosen].sum())
            if chosen.any():
                di = i[chosen, None] - wells[None, :, 0]
                dj = j[chosen, None] - wells[None, :, 1]
                dist = np.sqrt(di * di + dj * dj) * 50.0
                nearest = dist.min(axis=1)
                out[f"interference_distance_{key}mpa_p50_m"] = float(np.quantile(nearest, 0.50))
                out[f"interference_distance_{key}mpa_p90_m"] = float(np.quantile(nearest, 0.90))
                out[f"interference_distance_{key}mpa_max_m"] = float(nearest.max())
                out[f"connected_cells_{key}mpa"] = int(chosen.sum())
            else:
                out[f"interference_distance_{key}mpa_p50_m"] = np.nan
                out[f"interference_distance_{key}mpa_p90_m"] = np.nan
                out[f"interference_distance_{key}mpa_max_m"] = np.nan
                out[f"connected_cells_{key}mpa"] = 0
        return out


def main():
    data = np.load(RESULTS, allow_pickle=True)
    ok = np.asarray(data["success_mask"], dtype=bool)
    rows = np.flatnonzero(ok)
    sample = np.asarray(data["sample_index"], dtype=int)
    peak = pd.read_csv(PEAKS).set_index("sample_index")
    records = []
    started = time.time()
    for n, row in enumerate(rows, 1):
        sid = int(sample[row])
        existing = None
        if OUT.exists():
            try:
                old = pd.read_csv(OUT)
                existing = old.loc[old.sample_index == sid]
            except Exception:
                existing = None
        if existing is not None and len(existing):
            records.append(existing.iloc[0].to_dict())
            continue
        rec = {"sample_index": sid}
        try:
            run_dir = Path(str(data["run_dir"][row]))
            sr3s = sorted(run_dir.glob("*.sr3"))
            if not sr3s:
                raise FileNotFoundError(f"no sr3 in {run_dir}")
            rec.update(extract(sr3s[0], float(peak.loc[sid, "pavg_peak_year"]), data["prod_wells"][row]))
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = repr(exc)
        records.append(rec)
        if n % 5 == 0 or n == len(rows):
            pd.DataFrame(records).sort_values("sample_index").to_csv(OUT, index=False)
            STATE.write_text(json.dumps({"processed": n, "total": len(rows), "elapsed_s": time.time() - started, "updated": time.time()}, indent=2), encoding="utf-8")
            print(f"{n}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
    pd.DataFrame(records).sort_values("sample_index").to_csv(OUT, index=False)
    STATE.write_text(json.dumps({"processed": len(rows), "total": len(rows), "finished": True, "elapsed_s": time.time() - started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
