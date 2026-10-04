from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NI, NJ, NK = 268, 203, 200
INJ_K_START, INJ_K_END = 41, 80
PRD_K_START, PRD_K_END = 1, 15


def read_cmg_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    data = np.zeros((ni, nj, nk), dtype=np.float64)
    current_k = None
    current_j = None
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("**") and "K =" in line and "J =" in line:
                if current_k is not None and current_j is not None and values:
                    arr = np.asarray(values[:ni], dtype=np.float64)
                    if len(arr) == ni:
                        data[:, current_j - 1, current_k - 1] = arr
                try:
                    parts = line.replace(",", " ").split()
                    current_k = int(parts[parts.index("K") + 2])
                    current_j = int(parts[parts.index("J") + 2])
                    values = []
                except Exception:
                    current_k = None
                    current_j = None
                    values = []
                continue
            if line.startswith("**") or line.startswith("RESULTS") or line.startswith("PRES"):
                continue
            try:
                values.extend(float(v) for v in line.split())
            except ValueError:
                continue
    if current_k is not None and current_j is not None and values:
        arr = np.asarray(values[:ni], dtype=np.float64)
        if len(arr) == ni:
            data[:, current_j - 1, current_k - 1] = arr
    return data


def read_property_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        for raw in f:
            parts = raw.strip().split()
            if parts:
                vals.append(float(parts[0]))
    arr = np.asarray(vals[: ni * nj * nk], dtype=np.float64)
    if arr.size != ni * nj * nk:
        raise ValueError(f"{path} has {arr.size} values, expected {ni * nj * nk}")
    return arr.reshape((ni, nj, nk), order="F")


def minmax_norm(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    vals = np.asarray(arr[mask], dtype=np.float64)
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return out
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi <= lo:
        return out
    out[mask] = ((arr[mask] - lo) / (hi - lo)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build side-view fields without pressure-threshold gating."
    )
    parser.add_argument("--property-dir", default=r"F:\CODE1\property_CMG")
    parser.add_argument("--seeds", default=r"F:\CODE1\opt_seeds.npz")
    parser.add_argument(
        "--out",
        default=r"F:\CODE1\prescreen_maps\static_sideview_fields_no_pressure_threshold.npz",
    )
    args = parser.parse_args()

    prop = Path(args.property_dir)
    seeds_path = Path(args.seeds)
    out_path = Path(args.out)
    prefix = "2026_0421_1day_property "
    suffix = " 2025-Jan-01.txt"

    pressure = read_cmg_txt(prop / f"{prefix}Pressure{suffix}") * 1e-3
    thickness = read_cmg_txt(prop / f"{prefix}Grid Thickness{suffix}")
    mu_gas = read_cmg_txt(prop / f"{prefix}Gas Viscosity{suffix}")
    mu_water = read_cmg_txt(prop / f"{prefix}Water Viscosity{suffix}")
    perm = read_property_txt(prop / "property_260421_all_perm.txt")
    poro = read_property_txt(prop / "property_260421_all_poro.txt")

    seeds = np.load(seeds_path, allow_pickle=False)
    bound_mask = np.asarray(seeds["bound_mask"], dtype=bool)
    valid = (
        bound_mask[:, :, None]
        & np.isfinite(perm)
        & np.isfinite(poro)
        & np.isfinite(thickness)
        & (perm > 1e-8)
        & (poro > 1e-8)
        & (thickness > 0.0)
    )
    mu_gas_safe = np.where(mu_gas > 1e-10, mu_gas, np.nan)
    mu_water_safe = np.where(mu_water > 1e-10, mu_water, np.nan)
    khphi = np.where(valid, perm * thickness * poro, 0.0)
    mob_inj = np.where(valid, khphi / mu_gas_safe, 0.0)
    mob_prod = np.where(valid, khphi / mu_water_safe, 0.0)

    pressure_n = minmax_norm(pressure, valid & (pressure > 0.0))
    khphi_n = minmax_norm(khphi, valid)
    mob_inj_n = minmax_norm(mob_inj, valid)
    mob_prod_n = minmax_norm(mob_prod, valid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        static_inj_3d=mob_inj_n,
        static_prod_3d=mob_prod_n,
        static_khphi_3d=khphi_n,
        pressure_initial_3d=pressure_n,
        static_valid_3d=valid,
        bound_mask=bound_mask,
        INJ_K_START=np.array(INJ_K_START),
        INJ_K_END=np.array(INJ_K_END),
        PRD_K_START=np.array(PRD_K_START),
        PRD_K_END=np.array(PRD_K_END),
    )
    summary = {
        "out": str(out_path),
        "definition": "side-view static fields without pressure-threshold gating",
        "valid_cells": int(np.sum(valid)),
        "static_inj_range": [float(np.nanmin(mob_inj_n)), float(np.nanmax(mob_inj_n))],
        "static_prod_range": [float(np.nanmin(mob_prod_n)), float(np.nanmax(mob_prod_n))],
        "pressure_initial_range": [float(np.nanmin(pressure_n)), float(np.nanmax(pressure_n))],
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
