from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner


DEFAULT_LHS_NPZ = ROOT / "opt_seeds_lhs.npz"
DEFAULT_EXPORT_ROOT = ROOT / "opt_runs_lhs_doe_export"
DEFAULT_SIP_CANDIDATES = (
    ROOT / "cmg" / "2026_0421.sip",
    Path("F:/other/cmg/2026_0421.sip"),
)


def build_manifest(cases: list[dict], group_name: str) -> dict:
    return {
        "group_name": group_name,
        "n_cases": len(cases),
        "sample_indices": [int(item["sample_index"]) for item in cases],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a machine-ready LHS DOE group with manifest.json and run_group.ps1."
    )
    parser.add_argument("--lhs", default=str(DEFAULT_LHS_NPZ), help="path to LHS npz")
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--start", type=int, required=True, help="inclusive sample index")
    parser.add_argument("--stop", type=int, required=True, help="exclusive sample index")
    parser.add_argument("--sim-years", type=int, default=80)
    parser.add_argument(
        "--runner-template",
        default=str(DEFAULT_EXPORT_ROOT / "machine_5" / "run_group.ps1"),
        help="existing run_group.ps1 to copy into the exported group",
    )
    parser.add_argument(
        "--sip-template",
        default="",
        help="SIP file to copy into group_dir/cmg/2026_0421.sip; auto-detected if omitted",
    )
    args = parser.parse_args()

    lhs_path = Path(args.lhs)
    export_root = Path(args.export_root)
    group_dir = export_root / args.group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    lhs = np.load(lhs_path, allow_pickle=False)
    inj_all = lhs["inj_wells"]
    prod_all = lhs["prod_wells"]
    n_total = len(inj_all)

    start = max(0, int(args.start))
    stop = min(int(args.stop), n_total)
    if start >= stop:
        raise ValueError(f"empty range: start={start}, stop={stop}, total={n_total}")

    cmg_runner.OPT_RUNS_DIR = str(group_dir)
    cases: list[dict] = []
    for idx in range(start, stop):
        inj = [tuple(map(int, row)) for row in inj_all[idx]]
        prod = [tuple(map(int, row)) for row in prod_all[idx]]
        run_dir, dat_stem, inj_rates, q_caps, inj_shortfall = cmg_runner.make_run_dat(
            run_id=idx,
            inj_wells=inj,
            prod_wells=prod,
            sim_years=int(args.sim_years),
            gen_subdir=f"lhs_doe/case_{idx:04d}",
            local_rid=1,
        )
        dat_path = Path(run_dir) / f"{dat_stem}.dat"
        cases.append(
            {
                "sample_index": idx,
                "relative_dat_path": str(dat_path.relative_to(group_dir)).replace("/", "\\"),
                "inj_wells": [list(w) for w in inj],
                "prod_wells": [list(w) for w in prod],
                "inj_rates_surface_nm3_day": [float(v) for v in inj_rates],
                "inj_capacity_surface_nm3_day": [float(v) for v in q_caps],
                "inj_shortfall_surface_nm3_day": float(inj_shortfall),
            }
        )

    with open(group_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(build_manifest(cases, args.group_name), f, indent=2, ensure_ascii=False)

    runner_template = Path(args.runner_template)
    if not runner_template.exists():
        raise FileNotFoundError(f"runner template not found: {runner_template}")
    shutil.copy2(runner_template, group_dir / "run_group.ps1")

    sip_template = Path(args.sip_template) if args.sip_template else None
    if sip_template is None:
        sip_template = next((p for p in DEFAULT_SIP_CANDIDATES if p.exists()), None)
    if sip_template is None or not sip_template.exists():
        raise FileNotFoundError(
            "SIP template not found. Pass --sip-template or place 2026_0421.sip under F:/other/cmg."
        )
    sip_dir = group_dir / "cmg"
    sip_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sip_template, sip_dir / "2026_0421.sip")

    print(f"Export group   : {group_dir}")
    print(f"Sample indices : {start}..{stop - 1}")
    print(f"Cases          : {stop - start}")
    print(f"SIP file       : {sip_dir / '2026_0421.sip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
