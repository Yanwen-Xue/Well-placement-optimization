\
\
\
\
\
\
\


from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner
import objectives
from validation_tools.run_lhs_doe import flatten_layout


DEFAULT_LHS_NPZ = ROOT / "opt_seeds_lhs.npz"
DEFAULT_EXPORT_ROOT = ROOT / "opt_runs_lhs_doe_export"
DEFAULT_OUT = ROOT / "opt_runs_lhs_doe" / "results_current.npz"

CASE_RE = re.compile(r"case_(\d{4})", re.IGNORECASE)
PREFERRED_PARTS = (
    "machine_local_rerun_114115_compat",
    "machine_local_rerun_200",
    "machine_1_takeover",
    "machine_4_takeover",
    "machine_5_return",
    "machine_4_return",
    "machine_2_return",
    "machine_1_local",
    "machine_4",
    "machine_5",
    "machine_2_remote",
)


def candidate_priority(path: Path) -> tuple[int, float]:
    text = str(path).lower()
    backup_penalty = 100 if "interrupted_backup" in text or "\\backup" in text or "/backup" in text else 0
    missing_sr3_penalty = 50 if not (path / "opt_001.sr3").exists() else 0
    run_dir_bonus = -1 if path.name.lower() == "run_001" else 0
    for rank, marker in enumerate(PREFERRED_PARTS):
        if marker.lower() in text:
            return backup_penalty + missing_sr3_penalty + rank + run_dir_bonus, -path.stat().st_mtime
    return backup_penalty + missing_sr3_penalty + len(PREFERRED_PARTS) + run_dir_bonus, -path.stat().st_mtime


def find_case_dirs(export_root: Path, start: int, stop: int) -> dict[int, Path]:
    by_case: dict[int, list[Path]] = {}
    result_files = list(export_root.rglob("opt_001.sr3")) + list(export_root.rglob("opt_001.out")) + list(export_root.rglob("opt_001.dat"))
    for file_path in result_files:
        match = CASE_RE.search(str(file_path.parent))
        if not match:
            continue
        idx = int(match.group(1))
        if idx < start or idx >= stop:
            continue
        by_case.setdefault(idx, []).append(file_path.parent)

    selected: dict[int, Path] = {}
    for idx, dirs in by_case.items():
        selected[idx] = sorted(set(dirs), key=candidate_priority)[0]
    return selected


def stop_line(out_path: Path) -> str:
    if not out_path.exists():
        return ""
    last = ""
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "STOP GEM" in line:
                    last = line.strip()
    except OSError:
        return ""
    return last


def build_eval_result(
    idx: int,
    run_dir: Path,
    inj_wells: np.ndarray,
    prod_wells: np.ndarray,
    grid_bot_2d: np.ndarray,
    sim_years: int,
) -> dict:
    sr3_path = run_dir / "opt_001.sr3"
    out_path = run_dir / "opt_001.out"
    lock_path = run_dir / "opt_001.sr3.nw.lck"
    stop = stop_line(out_path)

    out = {
        "run_id": idx,
        "run_dir": str(run_dir),
        "inj_wells": [tuple(map(int, w)) for w in inj_wells],
        "prod_wells": [tuple(map(int, w)) for w in prod_wells],
        "sim_years": int(sim_years),
        "run_success": False,
        "success": False,
        "full_duration_reached": False,
        "time_days_final": float("nan"),
        "failure_reason": "missing_result",
        "dynamic_injection_feasible": True,
        "inj_rates_surface_nm3_day": np.full(5, np.nan),
        "inj_bhp_max_per_well": np.array([]),
    }

    if lock_path.exists():
        out["failure_reason"] = "running"
        return out
    if not sr3_path.exists() or sr3_path.stat().st_size <= 0:
        out["failure_reason"] = "missing_sr3"
        return out
    if "Data error" in stop or "Fatal" in stop or "Time step too small" in stop:
        out["failure_reason"] = "simulation_failed"
        return out

    try:
        sr3 = cmg_runner.parse_sr3(str(sr3_path))
    except Exception as exc:
        out["failure_reason"] = f"parse_error:{type(exc).__name__}"
        return out

    out["run_success"] = ("End of simulation" in stop) or (stop == "")
    out["time_days"] = sr3["time_days_sp"]
    out["pavg"] = sr3["pavg"]
    out["ghgthy"] = sr3["ghgthy"]
    out["time_days_final"] = (
        float(np.max(sr3["time_days"]))
        if sr3["time_days"] is not None and len(sr3["time_days"]) > 0
        else float("nan")
    )
    out["full_duration_reached"] = cmg_runner.reached_full_sim_duration(sr3["time_days"], sim_years)
    out["success"] = bool(out["run_success"] and out["full_duration_reached"])
    out["failure_reason"] = "" if out["success"] else "simulation_incomplete"

    ghgsol = sr3.get("ghgsol")
    ghgthy = sr3.get("ghgthy")
    ghgscrit = sr3.get("ghgscrit")
    out["ghgsol_final"] = float(ghgsol[-1]) if ghgsol is not None and len(ghgsol) > 0 else 0.0
    out["ghgthy_final"] = float(ghgthy[-1]) if ghgthy is not None and len(ghgthy) > 0 else 0.0
    out["ghgscrit_final"] = float(ghgscrit[-1]) if ghgscrit is not None and len(ghgscrit) > 0 else 0.0

    aqu = sr3.get("aqucommol_co2")
    gas = sr3.get("gascommol_co2")
    inj = sr3.get("gascmlisc_co2")
    out["aqu_initial_mol"] = float(aqu[0]) if aqu is not None and len(aqu) > 0 else 0.0
    out["aqu_final_mol"] = float(aqu[-1]) if aqu is not None and len(aqu) > 0 else 0.0
    out["gas_final_mol"] = float(gas[-1]) if gas is not None and len(gas) > 0 else 0.0
    out["inj_cum_mol"] = float(inj[-1]) if inj is not None and len(inj) > 0 else 0.0
    out["dissolved_net_mol"] = out["aqu_final_mol"] - out["aqu_initial_mol"]
    out["mobile_krg_fraction"] = (
        float(sr3["mobile_krg_fraction"])
        if sr3.get("mobile_krg_fraction") is not None
        else np.nan
    )
    out["plume_area_sg0p1_km2"] = (
        float(sr3["plume_area_sg0p1_km2"])
        if sr3.get("plume_area_sg0p1_km2") is not None
        else np.nan
    )
    out["plume_area_sg0p1_m2"] = (
        float(sr3["plume_area_sg0p1_m2"])
        if sr3.get("plume_area_sg0p1_m2") is not None
        else np.nan
    )
    out["plume_n_columns_sg0p1"] = (
        int(sr3["plume_n_columns_sg0p1"])
        if sr3.get("plume_n_columns_sg0p1") is not None
        else -1
    )
    out["plume_day"] = (
        float(sr3["plume_day"])
        if sr3.get("plume_day") is not None
        else np.nan
    )
    out["pressure_heterogeneity_injend_std_mpa"] = (
        float(sr3["pressure_heterogeneity_injend_std_mpa"])
        if sr3.get("pressure_heterogeneity_injend_std_mpa") is not None
        else np.nan
    )
    out["pressure_heterogeneity_injend_mean_mpa"] = (
        float(sr3["pressure_heterogeneity_injend_mean_mpa"])
        if sr3.get("pressure_heterogeneity_injend_mean_mpa") is not None
        else np.nan
    )

    prod_names = sorted(n for n in sr3["watvolsc"] if n.startswith("PRD"))
    out["prod_watvolsc_final_per_well"] = (
        np.array([float(sr3["watvolsc"][name][-1]) for name in prod_names], dtype=float)
        if prod_names
        else np.array([], dtype=float)
    )
    out["watvolsc_total"] = float(np.sum(out["prod_watvolsc_final_per_well"]))
    out["gasvolsc_total"] = sum(v[-1] for v in sr3["gasvolsc"].values()) if sr3["gasvolsc"] else 0.0

    inj_names = sorted(n for n in sr3["bhp"] if n.startswith("INJ"))
    if inj_names:
        out["inj_bhp_max_per_well"] = np.array([float(sr3["bhp"][name].max()) for name in inj_names])
        out["max_inj_bhp"] = float(out["inj_bhp_max_per_well"].max())
    else:
        out["inj_bhp_max_per_well"] = np.array([])
        out["max_inj_bhp"] = np.nan

    all_wells = out["inj_wells"] + out["prod_wells"]
    top_3d = cmg_runner._load_top_3d()
    inj_depths = [
        max(1.0, cmg_runner.SURF_ELEV + float(top_3d[wi, wj, cmg_runner.INJ_K_TOP - 1]))
        for (wi, wj) in out["inj_wells"]
    ]
    prd_depths = [
        max(1.0, cmg_runner.SURF_ELEV + float(top_3d[wi, wj, cmg_runner.PRD_K_TOP - 1]))
        for (wi, wj) in out["prod_wells"]
    ]
    out["well_depths_m"] = np.array(inj_depths + prd_depths)
    out["well_drill_depths_m"] = np.array(
        [max(1.0, cmg_runner.SURF_ELEV + grid_bot_2d[wi, wj]) for (wi, wj) in all_wells]
    )
    return out


def build_row(idx: int, run_dir: Path | None, lhs: dict, sim_years: int) -> dict:
    inj_wells = lhs["inj_wells"][idx].astype(np.int32)
    prod_wells = lhs["prod_wells"][idx].astype(np.int32)
    if run_dir is None:
        eval_result = {
            "run_id": idx,
            "run_dir": "",
            "inj_wells": [tuple(map(int, w)) for w in inj_wells],
            "prod_wells": [tuple(map(int, w)) for w in prod_wells],
            "success": False,
            "run_success": False,
            "full_duration_reached": False,
            "time_days_final": float("nan"),
            "failure_reason": "missing_result",
            "inj_rates_surface_nm3_day": np.full(5, np.nan),
            "inj_bhp_max_per_well": np.array([]),
        }
    else:
        eval_result = build_eval_result(
            idx=idx,
            run_dir=run_dir,
            inj_wells=inj_wells,
            prod_wells=prod_wells,
            grid_bot_2d=lhs["grid_bot_2d"],
            sim_years=sim_years,
        )

    F1, F2, F3, feasible = objectives.compute_objectives(
        eval_result,
        lhs["bound_mask"],
        float(lhs["dx"]),
        d_inj_inj_m=200.0,
        d_prod_prod_m=200.0,
        d_inj_prod_m=300.0,
    )
    assess = objectives.assess_eval_result(
        eval_result,
        lhs["bound_mask"],
        float(lhs["dx"]),
        d_inj_inj_m=200.0,
        d_prod_prod_m=200.0,
        d_inj_prod_m=300.0,
    )
    failure_reason = assess["failure_reason"]
    if failure_reason == objectives.FAIL_SIMULATION:
        failure_reason = str(eval_result.get("failure_reason") or failure_reason)

    inj_bhp = np.asarray(eval_result.get("inj_bhp_max_per_well", np.array([])), dtype=float)
    if len(inj_bhp) != 5:
        inj_bhp = np.full(5, np.nan)
    inj_rates = np.asarray(eval_result.get("inj_rates_surface_nm3_day", np.array([])), dtype=float)
    if len(inj_rates) != 5:
        inj_rates = np.full(5, np.nan)
    prod_wat = np.asarray(eval_result.get("prod_watvolsc_final_per_well", np.array([])), dtype=float)
    if len(prod_wat) != 5:
        prod_wat = np.full(5, np.nan)
    mobile_krg_fraction = float(eval_result.get("mobile_krg_fraction", np.nan))
    plume_area_sg0p1_km2 = float(eval_result.get("plume_area_sg0p1_km2", np.nan))
    plume_area_sg0p1_m2 = float(eval_result.get("plume_area_sg0p1_m2", np.nan))
    plume_n_columns_sg0p1 = int(eval_result.get("plume_n_columns_sg0p1", -1))
    plume_day = float(eval_result.get("plume_day", np.nan))
    pressure_heterogeneity_injend_std_mpa = float(
        eval_result.get("pressure_heterogeneity_injend_std_mpa", np.nan)
    )
    pressure_heterogeneity_injend_mean_mpa = float(
        eval_result.get("pressure_heterogeneity_injend_mean_mpa", np.nan)
    )

    return {
        "sample_index": idx,
        "x": flatten_layout(inj_wells, prod_wells),
        "objectives": np.array([F1, F2, F3], dtype=float),
        "success": bool(feasible),
        "failure_reason": failure_reason,
        "run_success": bool(eval_result.get("run_success", False)),
        "full_duration_reached": bool(eval_result.get("full_duration_reached", False)),
        "time_days_final": float(eval_result.get("time_days_final", float("nan"))),
        "expected_time_days": float(cmg_runner.expected_sim_days(sim_years)),
        "sim_years": int(sim_years),
        "bhp_feasible": 1 if failure_reason != objectives.FAIL_BHP and bool(feasible) else 0,
        "elapsed_s": float("nan"),
        "run_dir": str(eval_result.get("run_dir", "")),
        "inj_rates_surface_nm3_day": inj_rates,
        "inj_bhp_max_per_well": inj_bhp,
        "prod_watvolsc_final_per_well": prod_wat,
        "mobile_krg_fraction": mobile_krg_fraction,
        "plume_area_sg0p1_km2": plume_area_sg0p1_km2,
        "plume_area_sg0p1_m2": plume_area_sg0p1_m2,
        "plume_n_columns_sg0p1": plume_n_columns_sg0p1,
        "plume_day": plume_day,
        "pressure_heterogeneity_injend_std_mpa": pressure_heterogeneity_injend_std_mpa,
        "pressure_heterogeneity_injend_mean_mpa": pressure_heterogeneity_injend_mean_mpa,
        "inj_wells": inj_wells,
        "prod_wells": prod_wells,
    }


def write_results(out_path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["sample_index"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    np.savez_compressed(
        out_path,
        sample_index=np.array([r["sample_index"] for r in rows], dtype=np.int32),
        X=np.vstack([r["x"] for r in rows]).astype(float),
        objectives=np.vstack([r["objectives"] for r in rows]).astype(float),
        success_mask=np.array([r["success"] for r in rows], dtype=bool),
        failure_mask=np.array([not r["success"] for r in rows], dtype=bool),
        failure_reason=np.array([str(r["failure_reason"]) for r in rows], dtype="<U64"),
        run_success=np.array([r["run_success"] for r in rows], dtype=bool),
        full_duration_reached=np.array([r["full_duration_reached"] for r in rows], dtype=bool),
        time_days_final=np.array([r["time_days_final"] for r in rows], dtype=float),
        expected_time_days=np.array([r["expected_time_days"] for r in rows], dtype=float),
        sim_years=np.array([r["sim_years"] for r in rows], dtype=np.int32),
        bhp_feasible=np.array([r["bhp_feasible"] for r in rows], dtype=np.int8),
        elapsed_s=np.array([r["elapsed_s"] for r in rows], dtype=float),
        run_dir=np.array([r["run_dir"] for r in rows], dtype="<U320"),
        inj_rates_surface_nm3_day=np.vstack([r["inj_rates_surface_nm3_day"] for r in rows]).astype(float),
        inj_bhp_max_per_well=np.vstack([r["inj_bhp_max_per_well"] for r in rows]).astype(float),
        prod_watvolsc_final_per_well=np.vstack([r["prod_watvolsc_final_per_well"] for r in rows]).astype(float),
        mobile_krg_fraction=np.array([r["mobile_krg_fraction"] for r in rows], dtype=float),
        plume_area_sg0p1_km2=np.array([r["plume_area_sg0p1_km2"] for r in rows], dtype=float),
        plume_area_sg0p1_m2=np.array([r["plume_area_sg0p1_m2"] for r in rows], dtype=float),
        plume_n_columns_sg0p1=np.array([r["plume_n_columns_sg0p1"] for r in rows], dtype=np.int32),
        plume_day=np.array([r["plume_day"] for r in rows], dtype=float),
        pressure_heterogeneity_injend_std_mpa=np.array(
            [r["pressure_heterogeneity_injend_std_mpa"] for r in rows], dtype=float
        ),
        pressure_heterogeneity_injend_mean_mpa=np.array(
            [r["pressure_heterogeneity_injend_mean_mpa"] for r in rows], dtype=float
        ),
        inj_wells=np.array([r["inj_wells"] for r in rows], dtype=np.int32).reshape(n, 5, 2),
        prod_wells=np.array([r["prod_wells"] for r in rows], dtype=np.int32).reshape(n, 5, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LHS results.npz from imported CMG export folders.")
    parser.add_argument("--lhs", default=str(DEFAULT_LHS_NPZ))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=200)
    parser.add_argument("--sim-years", type=int, default=80)
    parser.add_argument("--max-cases", type=int, default=0, help="debug limit after scanning")
    args = parser.parse_args()

    lhs_npz = np.load(args.lhs, allow_pickle=True)
    lhs = {name: lhs_npz[name] for name in lhs_npz.files}
    stop = min(int(args.stop), len(lhs["inj_wells"]))
    case_dirs = find_case_dirs(Path(args.export_root), int(args.start), stop)

    rows = []
    indices = list(range(int(args.start), stop))
    if int(args.max_cases) > 0:
        indices = indices[: int(args.max_cases)]
    for count, idx in enumerate(indices, start=1):
        run_dir = case_dirs.get(idx)
        print(f"[{count:04d}/{len(indices):04d}] case_{idx:04d} dir={run_dir}", flush=True)
        rows.append(build_row(idx, run_dir, lhs, int(args.sim_years)))

    out_path = Path(args.out)
    write_results(out_path, rows)
    reasons = Counter(str(r["failure_reason"]) for r in rows)
    summary = {
        "lhs": str(args.lhs),
        "export_root": str(args.export_root),
        "out": str(out_path),
        "n_rows": len(rows),
        "n_success": int(sum(bool(r["success"]) for r in rows)),
        "n_failure": int(sum(not bool(r["success"]) for r in rows)),
        "failure_reason_counts": dict(reasons),
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results: {out_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
