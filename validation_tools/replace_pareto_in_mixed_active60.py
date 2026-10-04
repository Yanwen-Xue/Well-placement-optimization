from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner
from objectives import check_boundary, check_spacing
from validation_tools.generate_static_active60_and_export import (
    layout_key,
    norm_columns,
    split_layout,
    valid_layout,
)


RESULTS = ROOT / "opt_runs_lhs_doe" / "results_0_623_current3obj_from_import_parallel_20260813.npz"
SEEDS = ROOT / "opt_seeds.npz"
NSGA_DIR = ROOT / "opt_runs_static_triview_nsga3_20260813"
OLD_ROOT = ROOT / "opt_runs_lhs_doe_export" / "static_active60_0624_0683_20260813"
OUT_ROOT = ROOT / "opt_runs_lhs_doe_export" / "current_mixed_active60_0624_0683_20260813"
OUT_LHS = ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz"
START_INDEX = 624
N_NEW = 60
PARETO_IDS = list(range(672, 679))


def choose_pareto_points(existing_keys, bound_mask, dx, count=7):
    nsga = np.load(NSGA_DIR / "ckpt_final_population.npz", allow_pickle=True)
    xs = []
    objs = []
    seen = set()
    for x, obj in zip(nsga["full_pop"], nsga["full_objs"]):
        x = np.rint(np.asarray(x, dtype=float)).astype(np.int32)
        key = layout_key(x)
        if key in existing_keys or key in seen:
            continue
        if valid_layout(x, bound_mask, dx):
            seen.add(key)
            xs.append(x)
            objs.append(np.asarray(obj, dtype=float))
    xs = np.asarray(xs, dtype=np.int32)
    objs = np.asarray(objs, dtype=float)
    if len(xs) < count:
        raise RuntimeError(f"Only {len(xs)} usable current Pareto points.")

    chosen = []
    for objective_id in range(3):
        chosen.append(int(np.argmin(objs[:, objective_id])))
    chosen = list(dict.fromkeys(chosen))
    onorm = norm_columns(objs)
    while len(chosen) < count:
        remaining = [i for i in range(len(xs)) if i not in chosen]
        distances = cdist(onorm[remaining], onorm[chosen])
        chosen.append(remaining[int(np.argmax(np.min(distances, axis=1)))])
    return xs[chosen[:count]], objs[chosen[:count]], str(nsga["surrogate_path"].item())


def update_dat_files(group_dir, replacements):
    cmg_runner.OPT_RUNS_DIR = str(group_dir)
    for sample_id, (x, _) in replacements.items():
        inj, prod = split_layout(x)
        cmg_runner.make_run_dat(
            run_id=sample_id,
            inj_wells=[tuple(map(int, row)) for row in inj],
            prod_wells=[tuple(map(int, row)) for row in prod],
            sim_years=80,
            gen_subdir=f"lhs_doe/case_{sample_id:04d}",
            local_rid=1,
        )


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    shutil.copytree(OLD_ROOT, OUT_ROOT)

    results = np.load(RESULTS, allow_pickle=True)
    seeds = np.load(SEEDS, allow_pickle=False)
    x_existing = np.rint(np.asarray(results["X"], dtype=float)).astype(np.int32)
    existing_keys = {layout_key(x) for x in x_existing}
    new_x, new_obj, surrogate_path = choose_pareto_points(
        existing_keys, np.asarray(seeds["bound_mask"], dtype=bool), float(seeds["dx"])
    )

    replacements = dict(zip(PARETO_IDS, zip(new_x, new_obj)))
    for group_dir in OUT_ROOT.glob("machine_*_current_mixed_0624_0683"):
        pass
    group_dirs = {
        "machine_1": OUT_ROOT / "machine_1_selected_0624_0683",
        "machine_4": OUT_ROOT / "machine_4_selected_0624_0683",
        "machine_5": OUT_ROOT / "machine_5_selected_0624_0683",
    }
    for machine, group_dir in group_dirs.items():
        ids = [672, 673, 674, 675] if machine == "machine_1" else [676] if machine == "machine_4" else [677, 678]
        update_dat_files(group_dir, {sid: replacements[sid] for sid in ids})

    csv_path = OUT_ROOT / "selected_samples.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    by_id = {int(row["sample_index"]): row for row in rows}
    machine_for_id = {}
    for machine, group_dir in group_dirs.items():
        manifest = json.loads((group_dir / "manifest.json").read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            machine_for_id[int(case["sample_index"])] = machine
    for row in rows:
        row["machine"] = machine_for_id[int(row["sample_index"])]
    for sid, (x, obj) in replacements.items():
        inj, prod = split_layout(x)
        row = by_id[sid]
        row["category"] = "current_nsga3_pareto"
        row["pred_F1"] = str(float(obj[0]))
        row["pred_F2_plume_km2"] = str(float(obj[1]))
        row["pred_F3_pressure_cv"] = str(float(obj[2]))
        row["inj_wells_0based"] = json.dumps(inj.tolist())
        row["prod_wells_0based"] = json.dumps(prod.tolist())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for manifest_path in group_dirs.values():
        data = json.loads((manifest_path / "manifest.json").read_text(encoding="utf-8"))
        for case in data["cases"]:
            sid = int(case["sample_index"])
            if sid not in replacements:
                continue
            x, obj = replacements[sid]
            inj, prod = split_layout(x)
            case["category"] = "current_nsga3_pareto"
            case["inj_wells"] = [[int(i), int(j)] for i, j in inj]
            case["prod_wells"] = [[int(i), int(j)] for i, j in prod]
        (manifest_path / "manifest.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    all_x = np.concatenate(
        [
            x_existing,
            np.asarray(
                [
                    np.concatenate(
                        [np.asarray(json.loads(row["inj_wells_0based"])).reshape(-1),
                         np.asarray(json.loads(row["prod_wells_0based"])).reshape(-1)]
                    )
                    for row in rows
                ],
                dtype=np.int32,
            ),
        ]
    )
    payload = {name: seeds[name] for name in seeds.files}
    payload["inj_wells"] = all_x[:, :10].reshape(-1, 5, 2)
    payload["prod_wells"] = all_x[:, 10:].reshape(-1, 5, 2)
    payload["_n_samples"] = np.array(len(all_x), dtype=np.int32)
    payload["_source"] = np.array("current_mixed_active60_with_current_nsga3", dtype=object)
    np.savez_compressed(OUT_LHS, **payload)

    summary = json.loads((OUT_ROOT / "selection_summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "results": str(RESULTS),
            "nsga_dir": str(NSGA_DIR),
            "surrogate_path": surrogate_path,
            "selection_method": "original mixed sampling; only the 7 Pareto rows replaced with current NSGA-III points",
            "old_nsga_output_used": False,
            "current_nsga3_pareto_count": 7,
            "category_counts": {
                "global_maximin": 10,
                "F1_active": 8,
                "F2_active": 12,
                "F3_active": 18,
                "current_nsga3_pareto": 7,
                "contrast": 5,
            },
            "all_pareto_rows_are_current_nsga3": True,
            "out_root": str(OUT_ROOT),
            "out_lhs": str(OUT_LHS),
            "group_dirs": {machine: str(path) for machine, path in group_dirs.items()},
        }
    )
    (OUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
