

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner
import objectives


DEFAULT_LHS_NPZ = ROOT / "opt_seeds_lhs.npz"
DEFAULT_OUT_DIR = ROOT / "opt_runs_lhs_doe"


def flatten_layout(inj_wells: np.ndarray, prod_wells: np.ndarray) -> np.ndarray:
    x = np.zeros(20, dtype=float)
    x[:10] = np.asarray(inj_wells, dtype=float).reshape(-1)
    x[10:] = np.asarray(prod_wells, dtype=float).reshape(-1)
    return x


def _bhp_status(eval_result: dict) -> tuple[int, list[int]]:
    inj_bhp = eval_result.get("inj_bhp_max_per_well", np.array([]))
    depths = eval_result.get("well_depths_m", np.array([]))
    feasible, diag = objectives.check_bhp_constraint(inj_bhp, depths)
    if len(inj_bhp) < 5 or len(depths) < 5:
        return -1, []
    return (1 if feasible else 0), [int(v) for v in diag.get("violators", [])]


def evaluate_case(case: dict) -> dict:
    idx = int(case["sample_index"])
    inj = [tuple(map(int, w)) for w in case["inj_wells"]]
    prod = [tuple(map(int, w)) for w in case["prod_wells"]]
    result = cmg_runner.evaluate(
        run_id=idx,
        inj_wells=inj,
        prod_wells=prod,
        grid_top_2d=case["grid_top_2d"],
        sim_years=int(case["sim_years"]),
        cleanup=bool(case["cleanup"]),
        gen_subdir=f"lhs_doe/case_{idx:04d}",
        local_rid=1,
        grid_bot_2d=case["grid_bot_2d"],
    )
    result["inj_wells"] = inj
    result["prod_wells"] = prod

    F1, F2, F3, feasible = objectives.compute_objectives(
        result,
        case["bound_mask"],
        case["dx"],
        d_inj_inj_m=case["d_inj_inj_m"],
        d_prod_prod_m=case["d_prod_prod_m"],
        d_inj_prod_m=case["d_inj_prod_m"],
    )
    assess = objectives.assess_eval_result(
        result,
        case["bound_mask"],
        case["dx"],
        d_inj_inj_m=case["d_inj_inj_m"],
        d_prod_prod_m=case["d_prod_prod_m"],
        d_inj_prod_m=case["d_inj_prod_m"],
    )
    bhp_status, bhp_violators = _bhp_status(result)
    pressure_cv = objectives.compute_F3(
        result.get("pressure_heterogeneity_injend_std_mpa", np.nan),
        result.get("pressure_heterogeneity_injend_mean_mpa", np.nan),
    )
    if pressure_cv >= objectives.PENALTY_VALUE:
        pressure_cv = np.nan

    return {
        "sample_index": idx,
        "x": flatten_layout(case["inj_wells"], case["prod_wells"]),
        "objectives": np.array([F1, F2, F3], dtype=float),
        "success": bool(feasible),
        "failure_reason": assess["failure_reason"],
        "run_success": bool(result.get("run_success", False)),
        "full_duration_reached": bool(result.get("full_duration_reached", False)),
        "time_days_final": float(result.get("time_days_final", float("nan"))),
        "expected_time_days": float(cmg_runner.expected_sim_days(case["sim_years"])),
        "sim_years": int(case["sim_years"]),
        "bhp_feasible": int(bhp_status),
        "bhp_violators": np.array(bhp_violators, dtype=np.int32),
        "elapsed_s": float(result.get("elapsed_s", float("nan"))),
        "run_dir": str(result.get("run_dir", "")),
        "inj_rates_surface_nm3_day": np.asarray(
            result.get("inj_rates_surface_nm3_day", np.array([])), dtype=float
        ),
        "inj_bhp_max_per_well": np.asarray(
            result.get("inj_bhp_max_per_well", np.array([])), dtype=float
        ),
        "prod_watvolsc_final_per_well": np.asarray(
            result.get("prod_watvolsc_final_per_well", np.array([])), dtype=float
        ),
        "plume_area_sg0p1_km2": float(result.get("plume_area_sg0p1_km2", np.nan)),
        "plume_n_columns_sg0p1": int(result.get("plume_n_columns_sg0p1", -1)),
        "plume_status": str(result.get("plume_status", "")),
        "plume_day": float(result.get("plume_day", np.nan)),
        "pressure_heterogeneity_injend_std_mpa": float(
            result.get("pressure_heterogeneity_injend_std_mpa", np.nan)
        ),
        "pressure_heterogeneity_injend_mean_mpa": float(
            result.get("pressure_heterogeneity_injend_mean_mpa", np.nan)
        ),
        "pressure_heterogeneity_cv": float(pressure_cv),
    }


def build_case_list(lhs_npz: Path, sim_years: int, cleanup: bool,
                    start: int, stop: int | None) -> tuple[dict, list[dict]]:
    lhs = np.load(lhs_npz, allow_pickle=False)
    inj = lhs["inj_wells"]
    prod = lhs["prod_wells"]
    n_total = len(inj)
    stop_idx = n_total if stop is None else min(int(stop), n_total)
    start_idx = max(0, int(start))
    meta = {
        "bound_mask": lhs["bound_mask"].astype(bool),
        "grid_top_2d": lhs["grid_top_2d"],
        "grid_bot_2d": lhs["grid_bot_2d"],
        "dx": float(lhs["dx"]),
        "sim_years": int(sim_years),
        "cleanup": bool(cleanup),
        "d_inj_inj_m": 200.0,
        "d_prod_prod_m": 200.0,
        "d_inj_prod_m": 300.0,
        "n_total": n_total,
        "sample_start": start_idx,
        "sample_stop": stop_idx,
        "lhs_npz": str(lhs_npz),
    }
    cases = []
    for idx in range(start_idx, stop_idx):
        cases.append({
            **meta,
            "sample_index": idx,
            "inj_wells": inj[idx].astype(np.int32),
            "prod_wells": prod[idx].astype(np.int32),
        })
    return meta, cases


def write_results(out_path: Path, meta: dict, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["sample_index"])
    n = len(rows)
    if n == 0:
        raise ValueError("no DOE rows to save")

    failure_reason = np.array([str(r["failure_reason"]) for r in rows], dtype="<U32")
    run_dir = np.array([str(r["run_dir"]) for r in rows], dtype="<U260")
    np.savez_compressed(
        out_path,
        sample_index=np.array([r["sample_index"] for r in rows], dtype=np.int32),
        X=np.vstack([r["x"] for r in rows]).astype(float),
        objectives=np.vstack([r["objectives"] for r in rows]).astype(float),
        success_mask=np.array([r["success"] for r in rows], dtype=bool),
        failure_mask=np.array([not r["success"] for r in rows], dtype=bool),
        failure_reason=failure_reason,
        run_success=np.array([r["run_success"] for r in rows], dtype=bool),
        full_duration_reached=np.array([r["full_duration_reached"] for r in rows], dtype=bool),
        time_days_final=np.array([r["time_days_final"] for r in rows], dtype=float),
        expected_time_days=np.array([r["expected_time_days"] for r in rows], dtype=float),
        sim_years=np.array([r["sim_years"] for r in rows], dtype=np.int32),
        bhp_feasible=np.array([r["bhp_feasible"] for r in rows], dtype=np.int8),
        elapsed_s=np.array([r["elapsed_s"] for r in rows], dtype=float),
        run_dir=run_dir,
        inj_rates_surface_nm3_day=np.vstack([r["inj_rates_surface_nm3_day"] for r in rows]).astype(float),
        inj_bhp_max_per_well=np.vstack([
            r["inj_bhp_max_per_well"] if len(r["inj_bhp_max_per_well"]) == 5 else np.full(5, np.nan)
            for r in rows
        ]).astype(float),
        prod_watvolsc_final_per_well=np.vstack([
            r["prod_watvolsc_final_per_well"] if len(r["prod_watvolsc_final_per_well"]) == 5 else np.full(5, np.nan)
            for r in rows
        ]).astype(float),
        plume_area_sg0p1_km2=np.array([r["plume_area_sg0p1_km2"] for r in rows], dtype=float),
        plume_n_columns_sg0p1=np.array([r["plume_n_columns_sg0p1"] for r in rows], dtype=np.int32),
        plume_status=np.array([r["plume_status"] for r in rows], dtype="<U64"),
        plume_day=np.array([r["plume_day"] for r in rows], dtype=float),
        pressure_heterogeneity_injend_std_mpa=np.array(
            [r["pressure_heterogeneity_injend_std_mpa"] for r in rows], dtype=float
        ),
        pressure_heterogeneity_injend_mean_mpa=np.array(
            [r["pressure_heterogeneity_injend_mean_mpa"] for r in rows], dtype=float
        ),
        pressure_heterogeneity_cv=np.array([r["pressure_heterogeneity_cv"] for r in rows], dtype=float),
        inj_wells=np.array([meta_case["inj_wells"] for meta_case in rows], dtype=np.int32).reshape(n, 5, 2),
        prod_wells=np.array([meta_case["prod_wells"] for meta_case in rows], dtype=np.int32).reshape(n, 5, 2),
    )


def write_summary(summary_path: Path, meta: dict, rows: list[dict]) -> None:
    reasons = Counter(r["failure_reason"] for r in rows)
    success = sum(1 for r in rows if r["success"])
    report = {
        "lhs_npz": str(meta.get("lhs_npz", DEFAULT_LHS_NPZ)),
        "n_rows": len(rows),
        "n_success": success,
        "n_failure": len(rows) - success,
        "success_rate": (success / len(rows)) if rows else 0.0,
        "sim_years": int(meta["sim_years"]),
        "expected_time_days": float(cmg_runner.expected_sim_days(meta["sim_years"])),
        "failure_reason_counts": dict(reasons),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> int:
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Run LHS DOE and build surrogate training labels.")
    parser.add_argument("--lhs", default=str(DEFAULT_LHS_NPZ), help="path to opt_seeds_lhs.npz")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="output directory")
    parser.add_argument("--parallel", type=int, default=8, help="number of concurrent CMG workers")
    parser.add_argument("--parasol", type=int, default=16, help="CMG PARASOL threads per worker")
    parser.add_argument("--sim-years", type=int, default=80, help="must match surrogate target horizon")
    parser.add_argument("--start", type=int, default=0, help="inclusive starting sample index")
    parser.add_argument("--stop", type=int, default=None, help="exclusive ending sample index")
    parser.add_argument("--cleanup", action="store_true", help="delete CMG run folders after each evaluation")
    args = parser.parse_args()

    lhs_path = Path(args.lhs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.npz"
    summary_path = out_dir / "summary.json"

    os.environ["CMG_PARASOL"] = str(args.parasol)
    cmg_runner.CMG_PARASOL = int(args.parasol)

    meta, cases = build_case_list(
        lhs_npz=lhs_path,
        sim_years=int(args.sim_years),
        cleanup=bool(args.cleanup),
        start=int(args.start),
        stop=args.stop,
    )
    print(
        f"LHS DOE: {len(cases)} samples from {lhs_path} | "
        f"sim_years={args.sim_years} | parallel={args.parallel} | parasol={args.parasol}"
    )

    rows = []
    with ProcessPoolExecutor(max_workers=int(args.parallel)) as ex:
        futures = [ex.submit(evaluate_case, case) for case in cases]
        for k, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append({
                **row,
                "inj_wells": cases[row["sample_index"] - meta["sample_start"]]["inj_wells"],
                "prod_wells": cases[row["sample_index"] - meta["sample_start"]]["prod_wells"],
            })
            print(
                f"[{k:04d}/{len(cases):04d}] idx={row['sample_index']:04d} "
                f"success={int(row['success'])} reason={row['failure_reason']} "
                f"days={row['time_days_final']:.1f}"
            )

    write_results(results_path, meta, rows)
    write_summary(summary_path, meta, rows)
    print(f"Saved DOE dataset: {results_path}")
    print(f"Saved summary    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
