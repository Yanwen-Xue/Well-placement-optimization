from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_tools import build_lhs_results_from_export as builder

LHS_PATH = ROOT / "opt_seeds_0_543_directLHS_replacement_344_543_20260723.npz"
EXPORT_ROOT = (
    ROOT
    / "opt_runs_lhs_doe_export"
    / "returned_results_inbox"
    / "machine_2_return"
    / "machine_2_30"
    / "lhs_doe"
)
OUT = ROOT / "opt_runs_lhs_doe" / "results_0_543_current3obj_from_import_parallel_20260809.npz"
SUMMARY = ROOT / "opt_runs_lhs_doe" / "results_0_543_current3obj_from_import_parallel_20260809.summary.json"

START = 0
STOP = 544
SIM_YEARS = 80
MAX_WORKERS = 8


def _failed_worker_row(idx: int, reason: str) -> dict:
    return {
        "sample_index": idx,
        "x": np.full(20, np.nan),
        "objectives": np.full(3, 1e9),
        "success": False,
        "failure_reason": reason,
        "run_success": False,
        "full_duration_reached": False,
        "time_days_final": float("nan"),
        "expected_time_days": float("nan"),
        "sim_years": SIM_YEARS,
        "bhp_feasible": 0,
        "elapsed_s": float("nan"),
        "run_dir": "",
        "inj_rates_surface_nm3_day": np.full(5, np.nan),
        "inj_bhp_max_per_well": np.full(5, np.nan),
        "prod_watvolsc_final_per_well": np.full(5, np.nan),
        "mobile_krg_fraction": np.nan,
        "plume_area_sg0p1_km2": np.nan,
        "plume_area_sg0p1_m2": np.nan,
        "plume_n_columns_sg0p1": -1,
        "plume_day": np.nan,
        "pressure_heterogeneity_injend_std_mpa": np.nan,
        "pressure_heterogeneity_injend_mean_mpa": np.nan,
        "inj_wells": np.full((5, 2), -1),
        "prod_wells": np.full((5, 2), -1),
    }


def worker(idx: int, run_dir_str: str) -> tuple[int, dict | None, str | None]:
    try:
        lhs_npz = np.load(LHS_PATH, allow_pickle=True)
        lhs = {name: lhs_npz[name] for name in lhs_npz.files}
        run_dir = Path(run_dir_str) if run_dir_str else None
        return idx, builder.build_row(idx, run_dir, lhs, SIM_YEARS), None
    except Exception:
        return idx, None, traceback.format_exc()


def main() -> int:
    case_dirs = builder.find_case_dirs(EXPORT_ROOT, START, STOP)
    tasks = [(idx, str(case_dirs[idx]) if idx in case_dirs else "") for idx in range(START, STOP)]

    rows: list[dict] = []
    errors: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, idx, run_dir) for idx, run_dir in tasks]
        for done, fut in enumerate(as_completed(futures), start=1):
            idx, row, err = fut.result()
            if err:
                errors[idx] = err
                row = _failed_worker_row(idx, "worker_error")
            rows.append(row)
            if done == 1 or done % 25 == 0 or done == len(tasks):
                print(f"completed {done}/{len(tasks)}", flush=True)

    builder.write_results(OUT, rows)
    reasons = Counter(str(r["failure_reason"]) for r in rows)
    summary = {
        "lhs": str(LHS_PATH),
        "export_root": str(EXPORT_ROOT),
        "out": str(OUT),
        "n_rows": len(rows),
        "n_success": int(sum(bool(r["success"]) for r in rows)),
        "n_failure": int(sum(not bool(r["success"]) for r in rows)),
        "failure_reason_counts": dict(reasons),
        "worker_errors": {str(k): v.splitlines()[-1] for k, v in errors.items()},
        "objective_names": [
            "F1_pressure_rise",
            "F2_plume_area_sg0p1_km2",
            "F3_pressure_CV_std_over_mean",
        ],
        "sg_threshold": 0.1,
        "plume_area_cell_km2": 0.0025,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
