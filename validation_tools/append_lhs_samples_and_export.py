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

import lhs_sampler
import cmg_runner


DEFAULT_BASE_LHS = ROOT / "opt_seeds_lhs.npz"
DEFAULT_OUT_LHS = ROOT / "opt_seeds_lhs_extended.npz"
DEFAULT_EXPORT_ROOT = ROOT / "opt_runs_lhs_doe_export"


def case_key(inj: np.ndarray, prod: np.ndarray) -> tuple[int, ...]:
    return tuple(np.concatenate([inj.reshape(-1), prod.reshape(-1)]).astype(int).tolist())


def build_manifest(cases: list[dict], group_name: str) -> dict:
    return {
        "group_name": group_name,
        "n_cases": len(cases),
        "sample_indices": [int(item["sample_index"]) for item in cases],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append new constraint-aware LHS samples after existing indices and export dat files."
    )
    parser.add_argument("--base-lhs", default=str(DEFAULT_BASE_LHS))
    parser.add_argument("--out-lhs", default=str(DEFAULT_OUT_LHS))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--group-name", default="machine_5_extra")
    parser.add_argument("--n-extra", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--oversample", type=int, default=30)
    parser.add_argument("--sim-years", type=int, default=80)
    args = parser.parse_args()

    base_lhs_path = Path(args.base_lhs)
    out_lhs_path = Path(args.out_lhs)
    export_root = Path(args.export_root)
    group_dir = export_root / args.group_name
    group_lhs_dir = group_dir / "lhs_doe"
    export_root.mkdir(parents=True, exist_ok=True)
    group_lhs_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(base_lhs_path, allow_pickle=True)
    base_inj = base["inj_wells"].astype(np.int32)
    base_prod = base["prod_wells"].astype(np.int32)
    base_n = len(base_inj)

    existing = {
        case_key(base_inj[i], base_prod[i])
        for i in range(base_n)
    }

    NI = int(base["NI"])
    NJ = int(base["NJ"])
    dx = float(base["dx"])
    d_bound_grids = max(1, int(np.ceil(lhs_sampler.D_BOUND_M / dx)))
    bound_mask = lhs_sampler.build_bound_mask(NI, NJ, d_bound_grids)

    new_inj = []
    new_prod = []
    batch_seed = int(args.seed)
    while len(new_inj) < int(args.n_extra):
        n_need = int(args.n_extra) - len(new_inj)
        n_target = max(n_need * 8, 200)
        cand_inj, cand_prod, _ = lhs_sampler.sample_constraint_aware_lhs(
            n_target=n_target,
            seed=batch_seed,
            oversample=int(args.oversample),
            NI=NI,
            NJ=NJ,
            dx=dx,
            bound_mask=bound_mask,
        )
        batch_seed += 1
        for inj, prod in zip(cand_inj, cand_prod):
            key = case_key(inj, prod)
            if key in existing:
                continue
            existing.add(key)
            new_inj.append(inj.astype(np.int32))
            new_prod.append(prod.astype(np.int32))
            if len(new_inj) >= int(args.n_extra):
                break

    new_inj_arr = np.array(new_inj, dtype=np.int32)
    new_prod_arr = np.array(new_prod, dtype=np.int32)
    all_inj = np.concatenate([base_inj, new_inj_arr], axis=0)
    all_prod = np.concatenate([base_prod, new_prod_arr], axis=0)

    payload = {name: base[name] for name in base.files if name not in {"inj_wells", "prod_wells", "_n_samples"}}
    payload["inj_wells"] = all_inj
    payload["prod_wells"] = all_prod
    payload["_n_samples"] = np.array(len(all_inj))
    np.savez_compressed(out_lhs_path, **payload)

    cmg_runner.OPT_RUNS_DIR = str(group_dir)
    exported_cases: list[dict] = []
    start_idx = base_n
    for offset in range(int(args.n_extra)):
        idx = start_idx + offset
        inj = [tuple(map(int, row)) for row in all_inj[idx]]
        prod = [tuple(map(int, row)) for row in all_prod[idx]]
        run_dir, dat_stem, inj_rates, q_caps, inj_shortfall = cmg_runner.make_run_dat(
            run_id=idx,
            inj_wells=inj,
            prod_wells=prod,
            sim_years=int(args.sim_years),
            gen_subdir=f"lhs_doe/case_{idx:04d}",
            local_rid=1,
        )
        dat_path = Path(run_dir) / f"{dat_stem}.dat"
        exported_cases.append(
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

    manifest = build_manifest(exported_cases, args.group_name)
    with open(group_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    shutil.copy2(export_root / "machine_5" / "run_group.ps1", group_dir / "run_group.ps1")

    with open(group_dir / "lhs_append_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_lhs": str(base_lhs_path),
                "out_lhs": str(out_lhs_path),
                "group_name": args.group_name,
                "base_n": base_n,
                "n_extra": int(args.n_extra),
                "sample_index_start": start_idx,
                "sample_index_stop": start_idx + int(args.n_extra),
                "seed": int(args.seed),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Base samples   : {base_n}")
    print(f"Extra samples  : {int(args.n_extra)}")
    print(f"Extended npz   : {out_lhs_path}")
    print(f"Export group   : {group_dir}")
    print(f"Sample indices : {start_idx}..{start_idx + int(args.n_extra) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
