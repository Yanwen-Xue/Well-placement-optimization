from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner
import lhs_sampler
from objectives import check_boundary, check_spacing
from validation_tools.train_surrogate import build_feature_extractor


RESULTS = ROOT / "opt_runs_lhs_doe" / "results_0_623_current3obj_from_import_parallel_20260813.npz"
SEEDS = ROOT / "opt_seeds.npz"
NSGA_DIR = ROOT / "opt_runs_mixed_active80_nsga3_20260809"
OUT_LHS = ROOT / "opt_seeds_0_683_static_active60_20260813.npz"
OUT_ROOT = ROOT / "opt_runs_lhs_doe_export" / "static_active60_0624_0683_20260813"
SIP = ROOT / "cmg" / "2026_0421.sip"

N_CAND_LHS = 12000
RANDOM_SEED = 20260813
N_MODELS = 6

# Rounded to the actual throughput and parallelism of the three machines.
MACHINE_COUNTS = {
    "machine_1": 33,
    "machine_4": 10,
    "machine_5": 17,
}

MACHINE_CATEGORY_COUNTS = {
    "machine_1": {
        "global_maximin": 5,
        "F1_active": 4,
        "F2_active": 7,
        "F3_active": 10,
        "pareto": 4,
        "contrast": 3,
    },
    "machine_4": {
        "global_maximin": 2,
        "F1_active": 1,
        "F2_active": 2,
        "F3_active": 3,
        "pareto": 1,
        "contrast": 1,
    },
    "machine_5": {
        "global_maximin": 3,
        "F1_active": 3,
        "F2_active": 3,
        "F3_active": 5,
        "pareto": 2,
        "contrast": 1,
    },
}

EXTERNAL_CATEGORY_COUNTS = {
    "machine_1": {
        "global_maximin": 1,
        "F1_active": 1,
        "F2_active": 1,
        "F3_active": 1,
        "pareto": 1,
        "contrast": 1,
    },
    "machine_4": {
        "F2_active": 1,
        "F3_active": 1,
    },
    "machine_5": {
        "global_maximin": 1,
        "F1_active": 1,
        "F3_active": 1,
        "pareto": 1,
    },
}

QUOTAS = {
    "global_maximin": 10,
    "F1_active": 8,
    "F2_active": 12,
    "F3_active": 18,
    "pareto": 7,
    "contrast": 5,
}


def flatten_layout(inj, prod):
    return np.concatenate([np.asarray(inj).reshape(-1), np.asarray(prod).reshape(-1)]).astype(float)


def split_layout(x):
    arr = np.rint(np.asarray(x, dtype=float)).astype(np.int32)
    return arr[:10].reshape(5, 2), arr[10:].reshape(5, 2)


def layout_key(x):
    return tuple(np.rint(np.asarray(x, dtype=float)).astype(int).tolist())


def valid_layout(x, bound_mask, dx):
    inj, prod = split_layout(x)
    return check_boundary(inj, prod, bound_mask) and check_spacing(
        inj,
        prod,
        lhs_sampler.D_INJ_INJ_M,
        lhs_sampler.D_PROD_PROD_M,
        lhs_sampler.D_INJ_PROD_M,
        dx,
    )


def norm_columns(arr):
    arr = np.asarray(arr, dtype=float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    den = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / den


def train_ensemble(X, y, pressure_std, pressure_mean):
    params = dict(
        n_estimators=800,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        min_child_weight=3.0,
        n_jobs=16,
        tree_method="hist",
        device="cpu",
    )
    rng = np.random.default_rng(RANDOM_SEED)
    models = []
    targets = [
        y[:, 0],
        y[:, 1],
        np.log(np.maximum(pressure_std, 1.0e-12)),
        np.log(np.maximum(pressure_mean, 1.0e-12)),
    ]
    for model_id in range(N_MODELS):
        sample_idx = rng.integers(0, len(X), size=len(X))
        heads = []
        for head_id, target in enumerate(targets):
            model = XGBRegressor(**params, random_state=RANDOM_SEED + 100 * model_id + head_id)
            model.fit(X[sample_idx], target[sample_idx])
            heads.append(model)
        models.append(heads)
    return models


def predict_ensemble(models, X):
    all_pred = []
    for f1, f2, std_model, mean_model in models:
        std = np.exp(std_model.predict(X))
        mean = np.exp(mean_model.predict(X))
        all_pred.append(
            np.column_stack(
                [
                    f1.predict(X),
                    f2.predict(X),
                    std / np.maximum(mean, 1.0e-12),
                ]
            )
        )
    all_pred = np.asarray(all_pred, dtype=float)
    return np.mean(all_pred, axis=0), np.std(all_pred, axis=0)


def nondominated_mask(objs):
    keep = np.ones(len(objs), dtype=bool)
    for i in range(len(objs)):
        if not keep[i]:
            continue
        dominates_i = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return keep


def greedy_pick(order, count, xnorm, selected, used):
    picked = []
    selected = list(selected)
    base = xnorm[selected] if selected else np.empty((0, xnorm.shape[1]))
    for idx in np.asarray(order, dtype=int):
        if idx in used:
            continue
        if len(base):
            if float(np.min(cdist(xnorm[[idx]], base))) < 0.035:
                continue
        picked.append(int(idx))
        used.add(int(idx))
        selected.append(int(idx))
        base = xnorm[selected]
        if len(picked) >= count:
            break
    return picked


def load_nsga_layouts(bound_mask, dx, existing_keys):
    out = []
    path = NSGA_DIR / "ckpt_final_population.npz"
    if not path.exists():
        return out
    data = np.load(path, allow_pickle=True)
    for name in ("full_pop", "representative_pop", "pop"):
        if name not in data.files:
            continue
        for x in data[name]:
            x = np.rint(np.asarray(x, dtype=float)).astype(np.int32)
            key = layout_key(x)
            if key in existing_keys:
                continue
            if valid_layout(x, bound_mask, dx):
                out.append(x)
    unique = {}
    for x in out:
        unique.setdefault(layout_key(x), x)
    return list(unique.values())


def build_export_group(group_name, entries, template):
    group_dir = OUT_ROOT / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "cmg").mkdir(parents=True, exist_ok=True)
    cmg_runner.OPT_RUNS_DIR = str(group_dir)
    cases = []
    for sample_index, x, category, role in entries:
        inj, prod = split_layout(x)
        inj = [tuple(map(int, row)) for row in inj]
        prod = [tuple(map(int, row)) for row in prod]
        run_dir, dat_stem, inj_rates, q_caps, inj_shortfall = cmg_runner.make_run_dat(
            run_id=int(sample_index),
            inj_wells=inj,
            prod_wells=prod,
            sim_years=80,
            gen_subdir=f"lhs_doe/case_{int(sample_index):04d}",
            local_rid=1,
        )
        dat_path = Path(run_dir) / f"{dat_stem}.dat"
        cases.append(
            {
                "sample_index": int(sample_index),
                "category": category,
                "role": role,
                "relative_dat_path": str(dat_path.relative_to(group_dir)).replace("/", "\\"),
                "inj_wells": [list(w) for w in inj],
                "prod_wells": [list(w) for w in prod],
                "inj_rates_surface_nm3_day": [float(v) for v in inj_rates],
                "inj_capacity_surface_nm3_day": [float(v) for v in q_caps],
                "inj_shortfall_surface_nm3_day": float(inj_shortfall),
            }
        )
    (group_dir / "manifest.json").write_text(
        json.dumps(
            {
                "group_name": group_name,
                "n_cases": len(cases),
                "sample_indices": [c["sample_index"] for c in cases],
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    shutil.copy2(template, group_dir / "run_group.ps1")
    shutil.copy2(SIP, group_dir / "cmg" / "2026_0421.sip")
    return group_dir


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = np.load(RESULTS, allow_pickle=True)
    seeds = np.load(SEEDS, allow_pickle=False)
    bound_mask = np.asarray(seeds["bound_mask"], dtype=bool)
    dx = float(seeds["dx"])

    X_all = np.rint(np.asarray(results["X"], dtype=float)).astype(np.int32)
    existing_keys = {layout_key(x) for x in X_all}
    rng = np.random.default_rng(RANDOM_SEED)

    lhs_inj, lhs_prod, lhs_stats = lhs_sampler.sample_constraint_aware_lhs(
        n_target=N_CAND_LHS,
        seed=RANDOM_SEED,
        oversample=4,
        NI=int(seeds["NI"]),
        NJ=int(seeds["NJ"]),
        dx=dx,
        bound_mask=bound_mask,
    )
    lhs_candidates = [
        np.rint(np.concatenate([inj.reshape(-1), prod.reshape(-1)])).astype(np.int32)
        for inj, prod in zip(lhs_inj, lhs_prod)
    ]
    nsga_candidates = load_nsga_layouts(bound_mask, dx, existing_keys)
    all_candidates = lhs_candidates + nsga_candidates
    unique = {}
    for x in all_candidates:
        key = layout_key(x)
        if key in existing_keys or key in unique:
            continue
        unique[key] = x
    Xcand_raw = np.asarray(list(unique.values()), dtype=np.float32)

    finite = (
        np.asarray(results["success_mask"], dtype=bool)
        & np.isfinite(results["objectives"]).all(axis=1)
        & np.isfinite(results["pressure_heterogeneity_injend_std_mpa"])
        & np.isfinite(results["pressure_heterogeneity_injend_mean_mpa"])
        & (results["pressure_heterogeneity_injend_mean_mpa"] > 0.0)
    )
    Xtrain_raw = np.asarray(results["X"][finite], dtype=float)
    ytrain = np.asarray(results["objectives"][finite], dtype=float)
    pstd = np.asarray(results["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float)
    pmean = np.asarray(results["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float)

    extractor, feature_report = build_feature_extractor(
        SEEDS, "prescreen_geometry_local_static_well_triview"
    )
    Xtrain = np.hstack([Xtrain_raw, np.asarray(extractor(Xtrain_raw), dtype=float)])
    Xcand = np.hstack([Xcand_raw, np.asarray(extractor(Xcand_raw), dtype=float)])
    ensemble = train_ensemble(Xtrain, ytrain, pstd, pmean)
    pred_mean, pred_unc = predict_ensemble(ensemble, Xcand)

    existing_and_candidates = np.vstack([np.asarray(X_all, dtype=float), Xcand_raw])
    xnorm = norm_columns(existing_and_candidates)
    existing_norm = xnorm[: len(X_all)]
    cand_norm = xnorm[len(X_all) :]
    candidate_offset = len(X_all)
    used = set()
    selected_global = []
    selected_categories = {}

    def select(category, order, count):
        order = np.asarray(order, dtype=int)
        global_order = order + candidate_offset
        picked_global = greedy_pick(global_order, count, xnorm, selected_global, {candidate_offset + i for i in used})
        for gidx in picked_global:
            local = gidx - candidate_offset
            selected_global.append(gidx)
            used.add(local)
            selected_categories[local] = category

    distance_existing = cdist(cand_norm, existing_norm).min(axis=1)
    select("global_maximin", np.argsort(-distance_existing), QUOTAS["global_maximin"])

    unc_norm = norm_columns(pred_unc)
    pred_norm = norm_columns(pred_mean)
    f1_score = unc_norm[:, 0] + 0.25 * np.abs(pred_norm[:, 0] - np.median(pred_norm[:, 0]))
    f2_score = unc_norm[:, 1] + 0.25 * np.abs(pred_norm[:, 1] - np.median(pred_norm[:, 1]))
    f3_score = 1.25 * unc_norm[:, 2] + 0.35 * pred_norm[:, 2]
    select("F1_active", np.argsort(-f1_score), QUOTAS["F1_active"])
    select("F2_active", np.argsort(-f2_score), QUOTAS["F2_active"])
    select("F3_active", np.argsort(-f3_score), QUOTAS["F3_active"])

    nd = np.where(nondominated_mask(pred_mean))[0]
    if len(nd) == 0:
        nd = np.arange(len(pred_mean))
    select("pareto", nd[np.argsort(np.linalg.norm(pred_norm[nd], axis=1))], QUOTAS["pareto"])

    f1_mid = np.abs(pred_norm[:, 0] - np.median(pred_norm[:, 0]))
    f2_mid = np.abs(pred_norm[:, 1] - np.median(pred_norm[:, 1]))
    contrast = np.lexsort((f1_mid + f2_mid, -pred_norm[:, 2]))
    select("contrast", contrast, QUOTAS["contrast"])

    if len(selected_global) < 60:
        select("global_fill", np.argsort(-distance_existing), 60 - len(selected_global))
    if len(selected_global) != 60:
        raise RuntimeError(f"selected {len(selected_global)} candidates, expected 60")

    selected_local = [idx - candidate_offset for idx in selected_global]
    selected_x = Xcand_raw[selected_local].astype(np.int32)
    start_idx = len(X_all)

    roles = ["training_extension"] * 60
    selected_rows = []
    for rank, local in enumerate(selected_local):
        selected_rows.append(
            {
                "new_rank": rank + 1,
                "sample_index": start_idx + rank,
                "category": selected_categories.get(local, "unknown"),
                "role": roles[rank],
                "pred_F1": float(pred_mean[local, 0]),
                "pred_F2_plume_km2": float(pred_mean[local, 1]),
                "pred_F3_pressure_cv": float(pred_mean[local, 2]),
                "unc_F1": float(pred_unc[local, 0]),
                "unc_F2": float(pred_unc[local, 1]),
                "unc_F3": float(pred_unc[local, 2]),
                "inj_wells_0based": json.dumps(selected_x[rank, :10].reshape(5, 2).tolist()),
                "prod_wells_0based": json.dumps(selected_x[rank, 10:].reshape(5, 2).tolist()),
            }
        )

    all_inj = np.concatenate([X_all[:, :10].reshape(-1, 5, 2), selected_x[:, :10].reshape(-1, 5, 2)])
    all_prod = np.concatenate([X_all[:, 10:].reshape(-1, 5, 2), selected_x[:, 10:].reshape(-1, 5, 2)])
    payload = {name: seeds[name] for name in seeds.files}
    payload["inj_wells"] = all_inj.astype(np.int32)
    payload["prod_wells"] = all_prod.astype(np.int32)
    payload["_n_samples"] = np.array(len(all_inj))
    payload["_source"] = np.array("static_sideview_active60_20260813", dtype=object)
    np.savez_compressed(OUT_LHS, **payload)

    templates = {
        "machine_1": ROOT / "opt_runs_lhs_doe_export" / "mixed_active80_0544_0623_202609_placeholder" / "run_group.ps1",
        "machine_4": ROOT / "opt_runs_lhs_doe_export" / "mixed_active80_0544_0623_20260809_fair" / "machine_4_cases_0591_0604" / "run_group.ps1",
        "machine_5": ROOT / "opt_runs_lhs_doe_export" / "mixed_active80_0544_0623_20260809_fair" / "machine_5_cases_0605_0623" / "run_group.ps1",
    }
    templates["machine_1"] = (
        ROOT
        / "opt_runs_lhs_doe_export"
        / "mixed_active80_0544_0623_20260809_fair"
        / "machine_1_cases_0544_0590"
        / "run_group.ps1"
    )
    grouped = {machine: [] for machine in MACHINE_COUNTS}
    category_ranks = {}
    for rank, local in enumerate(selected_local):
        category = selected_categories.get(local, "unknown")
        category_ranks.setdefault(category, []).append(rank)
    for category, ranks in category_ranks.items():
        offset = 0
        for machine in ("machine_1", "machine_4", "machine_5"):
            count = MACHINE_CATEGORY_COUNTS[machine].get(category, 0)
            for rank in ranks[offset : offset + count]:
                grouped[machine].append(
                    (
                        start_idx + rank,
                        selected_x[rank],
                        category,
                        roles[rank],
                    )
                )
            offset += count
        if offset != len(ranks):
            raise RuntimeError(f"category allocation mismatch for {category}: {offset} != {len(ranks)}")
    for machine in grouped:
        grouped[machine].sort(key=lambda item: item[0])
        if len(grouped[machine]) != MACHINE_COUNTS[machine]:
            raise RuntimeError(
                f"machine allocation mismatch for {machine}: "
                f"{len(grouped[machine])} != {MACHINE_COUNTS[machine]}"
            )
        external_need = EXTERNAL_CATEGORY_COUNTS[machine]
        updated = []
        external_used = {key: 0 for key in external_need}
        for sample_index, x, category, _ in grouped[machine]:
            role = "training_extension"
            if external_used.get(category, 0) < external_need.get(category, 0):
                role = "external_validation"
                external_used[category] = external_used.get(category, 0) + 1
                roles[sample_index - start_idx] = role
            updated.append((sample_index, x, category, role))
        grouped[machine] = updated
        if external_used != external_need:
            raise RuntimeError(
                f"external-validation allocation mismatch for {machine}: "
                f"{external_used} != {external_need}"
            )

    for rank, role in enumerate(roles):
        selected_rows[rank]["role"] = role

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "selected_samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)
    group_dirs = {}
    for machine, entries in grouped.items():
        group_dirs[machine] = str(build_export_group(
            f"{machine}_selected_0624_0683",
            entries,
            templates[machine],
        ))

    counts = {key: 0 for key in QUOTAS}
    for row in selected_rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    summary = {
        "results": str(RESULTS),
        "seeds": str(SEEDS),
        "out_lhs": str(OUT_LHS),
        "n_existing": int(len(X_all)),
        "n_new": 60,
        "sample_index_start": start_idx,
        "sample_index_stop_inclusive": start_idx + 59,
        "n_candidate_lhs": int(len(lhs_candidates)),
        "n_nsga_candidates": int(len(nsga_candidates)),
        "n_unique_candidates": int(len(Xcand_raw)),
        "feature_mode": "prescreen_geometry_local_static_well_triview",
        "feature_report": feature_report,
        "category_counts": counts,
        "machine_counts": MACHINE_COUNTS,
        "group_dirs": group_dirs,
        "lhs_stats": lhs_stats,
        "roles": {
            "training_extension": 48,
            "external_validation": 12,
        },
    }
    (OUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
