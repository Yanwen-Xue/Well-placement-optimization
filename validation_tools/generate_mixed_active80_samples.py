from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lhs_sampler
from objectives import check_boundary, check_spacing
from validation_tools.train_surrogate import build_feature_extractor


BASE_LHS = ROOT / "opt_seeds_0_623_mixed_active80_20260809.npz"
RESULTS = ROOT / "opt_runs_lhs_doe" / "results_0_623_current3obj_from_import_parallel_20260813.npz"
NSGA_DIR = ROOT / "opt_runs_mixed_active80_nsga3_20260809"
OUT_LHS = ROOT / "opt_seeds_0_703_mixed_active80_official_20260813.npz"
OUT_DIR = ROOT / "opt_runs_lhs_doe_export" / "mixed_active80_0624_0703_official_20260813"
SUMMARY_CSV = OUT_DIR / "mixed_active80_selected_samples.csv"
SUMMARY_JSON = OUT_DIR / "mixed_active80_summary.json"

N_CAND_LHS = 30000
RANDOM_SEED = 20260809
QUOTAS = {
    "global_maximin": 15,
    "F1_active": 10,
    "F2_active": 10,
    "F3_active": 20,
    "nsga3_pareto": 15,
    "contrast": 10,
}


def flatten_layout(inj, prod):
    return np.concatenate([np.asarray(inj).reshape(-1), np.asarray(prod).reshape(-1)]).astype(float)


def split_layout(x):
    arr = np.rint(np.asarray(x, dtype=float)).astype(np.int32)
    return arr[:10].reshape(5, 2), arr[10:].reshape(5, 2)


def layout_key(inj, prod):
    return tuple(np.concatenate([np.asarray(inj).reshape(-1), np.asarray(prod).reshape(-1)]).astype(int).tolist())


def valid_layout(inj, prod, bound_mask, dx):
    return check_boundary(inj, prod, bound_mask) and check_spacing(
        inj,
        prod,
        lhs_sampler.D_INJ_INJ_M,
        lhs_sampler.D_PROD_PROD_M,
        lhs_sampler.D_INJ_PROD_M,
        dx,
    )


def add_pairwise_features(Xraw, Xbase):
    feats = []
    for x in Xraw:
        inj = x[:10].reshape(5, 2)
        prod = x[10:].reshape(5, 2)
        vals = []
        for group in (inj, prod):
            ds = [
                np.linalg.norm(group[i] - group[j]) * 50.0
                for i in range(5)
                for j in range(i + 1, 5)
            ]
            vals += [min(ds), float(np.mean(ds)), max(ds), float(np.std(ds))]
        ds = [np.linalg.norm(a - b) * 50.0 for a in inj for b in prod]
        vals += [min(ds), float(np.mean(ds)), max(ds), float(np.std(ds))]
        for group in (inj, prod):
            c = group.mean(axis=0)
            rr = np.linalg.norm((group - c) * 50.0, axis=1)
            vals += [float(rr.mean()), float(rr.max()), float(rr.std())]
        feats.append(vals)
    return np.hstack([Xbase, np.asarray(feats, dtype=float)])


def build_features(X, seeds_path):
    extractor, _ = build_feature_extractor(seeds_path, "prescreen_geometry_local")
    Xbase = np.hstack([X, np.asarray(extractor(X), dtype=float)])
    return add_pairwise_features(X, Xbase)


def train_ensemble(Xtrain, ytrain, pressure_std, pressure_mean, n_models=8):
    rng = np.random.default_rng(RANDOM_SEED)
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
    ensemble = []
    n = len(Xtrain)
    for m in range(n_models):
        idx = rng.choice(n, size=n, replace=True)
        models = []
        for target in (
            ytrain[:, 0],
            ytrain[:, 1],
            np.log(np.maximum(pressure_std, 1e-12)),
            np.log(np.maximum(pressure_mean, 1e-12)),
        ):
            model = XGBRegressor(**params, random_state=RANDOM_SEED + m)
            model.fit(Xtrain[idx], target[idx])
            models.append(model)
        ensemble.append(models)
    return ensemble


def predict_ensemble(ensemble, Xcand):
    preds = []
    for f1_model, f2_model, std_model, mean_model in ensemble:
        f1 = f1_model.predict(Xcand)
        f2 = f2_model.predict(Xcand)
        std = np.exp(std_model.predict(Xcand))
        mean = np.exp(mean_model.predict(Xcand))
        preds.append(np.column_stack([f1, f2, std / mean]))
    arr = np.asarray(preds, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def nondominated_mask(objs):
    n = len(objs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def norm_columns(arr):
    arr = np.asarray(arr, dtype=float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    den = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / den


def greedy_pick(order, n_pick, Xnorm, selected_idx, used_idx):
    picked = []
    selected_set = set(selected_idx)
    used_set = set(used_idx)
    base = Xnorm[list(selected_set)] if selected_set else np.empty((0, Xnorm.shape[1]))
    for idx in order:
        idx = int(idx)
        if idx in used_set:
            continue
        if base.size:
            mind = float(np.min(cdist(Xnorm[[idx]], base)))
            if mind < 0.035:
                continue
        picked.append(idx)
        used_set.add(idx)
        selected_set.add(idx)
        base = Xnorm[list(selected_set)]
        if len(picked) >= n_pick:
            break
    return picked


def load_nsga_candidates(base, bound_mask, dx):
    out = []
    ckpt = NSGA_DIR / "ckpt_final_population.npz"
    if ckpt.exists():
        data = np.load(ckpt, allow_pickle=True)
        for key in ("full_pop", "representative_pop", "pop"):
            if key not in data.files:
                continue
            for x in data[key]:
                inj, prod = split_layout(x)
                if valid_layout(inj, prod, bound_mask, dx):
                    out.append((inj, prod))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = np.load(BASE_LHS, allow_pickle=True)
    results = np.load(RESULTS, allow_pickle=True)
    bound_mask = np.asarray(base["bound_mask"], dtype=bool)
    dx = float(base["dx"])
    ni = int(base["NI"]) if "NI" in base.files else bound_mask.shape[0]
    nj = int(base["NJ"]) if "NJ" in base.files else bound_mask.shape[1]

    existing_inj = np.asarray(base["inj_wells"], dtype=np.int32)
    existing_prod = np.asarray(base["prod_wells"], dtype=np.int32)
    existing_keys = {layout_key(i, p) for i, p in zip(existing_inj, existing_prod)}
    existing_x = np.vstack([flatten_layout(i, p) for i, p in zip(existing_inj, existing_prod)])

    cand_inj, cand_prod, lhs_stats = lhs_sampler.sample_constraint_aware_lhs(
        n_target=N_CAND_LHS,
        seed=RANDOM_SEED,
        oversample=4,
        NI=ni,
        NJ=nj,
        dx=dx,
        bound_mask=bound_mask,
    )
    nsga_pairs = load_nsga_candidates(base, bound_mask, dx)
    all_pairs = [(cand_inj[i], cand_prod[i]) for i in range(len(cand_inj))] + nsga_pairs

    unique_pairs = []
    unique_keys = set()
    for inj, prod in all_pairs:
        key = layout_key(inj, prod)
        if key in existing_keys or key in unique_keys:
            continue
        unique_pairs.append((np.asarray(inj, dtype=np.int32), np.asarray(prod, dtype=np.int32)))
        unique_keys.add(key)

    Xcand = np.vstack([flatten_layout(i, p) for i, p in unique_pairs])
    Xexisting_norm = norm_columns(np.vstack([existing_x, Xcand]))[: len(existing_x)]
    Xcand_norm = norm_columns(np.vstack([existing_x, Xcand]))[len(existing_x) :]
    Xallnorm = np.vstack([Xexisting_norm, Xcand_norm])
    cand_offset = len(existing_x)

    success = np.asarray(results["success_mask"], dtype=bool)
    finite = (
        success
        & np.isfinite(results["objectives"]).all(axis=1)
        & np.isfinite(results["pressure_heterogeneity_injend_std_mpa"])
        & np.isfinite(results["pressure_heterogeneity_injend_mean_mpa"])
        & (results["pressure_heterogeneity_injend_mean_mpa"] > 0)
    )
    Xtrain_raw = np.asarray(results["X"][finite], dtype=float)
    ytrain = np.asarray(results["objectives"][finite], dtype=float)
    ptr_std = np.asarray(results["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float)
    ptr_mean = np.asarray(results["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float)

    Xtrain = build_features(Xtrain_raw, BASE_LHS)
    Xfeat_cand = build_features(Xcand, BASE_LHS)
    ensemble = train_ensemble(Xtrain, ytrain, ptr_std, ptr_mean, n_models=8)
    pred_mean, pred_std = predict_ensemble(ensemble, Xfeat_cand)

    selected = []
    selected_categories = {}
    used = set()
    selected_global = []

    def select(category, order, count):
        picked = greedy_pick(order, count, Xallnorm, selected_global, {cand_offset + u for u in used})
        for global_idx in picked:
            local_idx = global_idx - cand_offset
            selected.append(local_idx)
            used.add(local_idx)
            selected_global.append(global_idx)
            selected_categories[local_idx] = category

    min_dist_to_existing = cdist(Xcand_norm, Xexisting_norm).min(axis=1)
    select("global_maximin", np.argsort(-min_dist_to_existing) + cand_offset, QUOTAS["global_maximin"])

    f1_score = norm_columns(pred_std[:, [0]]).ravel() + 0.25 * norm_columns(np.abs(pred_mean[:, [0]] - np.median(ytrain[:, 0]))).ravel()
    f2_score = norm_columns(pred_std[:, [1]]).ravel() + 0.25 * norm_columns(np.abs(pred_mean[:, [1]] - np.median(ytrain[:, 1]))).ravel()
    f3_score = (
        1.25 * norm_columns(pred_std[:, [2]]).ravel()
        + 0.35 * norm_columns(np.abs(pred_mean[:, [2]] - np.median(ytrain[:, 2]))).ravel()
        + 0.20 * norm_columns(pred_mean[:, [2]]).ravel()
    )
    select("F1_active", np.argsort(-f1_score) + cand_offset, QUOTAS["F1_active"])
    select("F2_active", np.argsort(-f2_score) + cand_offset, QUOTAS["F2_active"])
    select("F3_active", np.argsort(-f3_score) + cand_offset, QUOTAS["F3_active"])

    nd = np.where(nondominated_mask(pred_mean))[0]
    if len(nd) == 0:
        nd = np.arange(len(Xcand))
    nd_obj = norm_columns(pred_mean[nd])
    ideal_dist = np.linalg.norm(nd_obj, axis=1)
    select("nsga3_pareto", nd[np.argsort(ideal_dist)] + cand_offset, QUOTAS["nsga3_pareto"])

    f1_mid = np.abs(pred_mean[:, 0] - np.median(ytrain[:, 0]))
    f2_mid = np.abs(pred_mean[:, 1] - np.median(ytrain[:, 1]))
    central = norm_columns(f1_mid[:, None]).ravel() + norm_columns(f2_mid[:, None]).ravel()
    high_f3_order = np.lexsort((central, -pred_mean[:, 2]))
    low_f3_order = np.lexsort((central, pred_mean[:, 2]))
    contrast_order = np.ravel(np.column_stack([high_f3_order[:200], low_f3_order[:200]]))
    select("contrast", contrast_order + cand_offset, QUOTAS["contrast"])

    if len(selected) < 80:
        fill_order = np.argsort(-min_dist_to_existing) + cand_offset
        select("global_fill", fill_order, 80 - len(selected))
    if len(selected) != 80:
        raise RuntimeError(f"selected {len(selected)} samples, expected 80")

    new_inj = np.asarray([unique_pairs[i][0] for i in selected], dtype=np.int32)
    new_prod = np.asarray([unique_pairs[i][1] for i in selected], dtype=np.int32)
    all_inj = np.concatenate([existing_inj, new_inj], axis=0)
    all_prod = np.concatenate([existing_prod, new_prod], axis=0)

    payload = {name: base[name] for name in base.files if name not in {"inj_wells", "prod_wells", "_n_samples"}}
    payload["inj_wells"] = all_inj
    payload["prod_wells"] = all_prod
    payload["_n_samples"] = np.array(len(all_inj))
    payload["_source"] = np.array("mixed_active80_official_log_f3_20260813", dtype=object)
    np.savez_compressed(OUT_LHS, **payload)

    start_idx = len(existing_inj)
    rows = []
    for rank, local_idx in enumerate(selected):
        rows.append(
            {
                "new_rank": rank + 1,
                "sample_index": start_idx + rank,
                "source_candidate_index": int(local_idx),
                "category": selected_categories[local_idx],
                "pred_F1": float(pred_mean[local_idx, 0]),
                "pred_F2_plume_km2": float(pred_mean[local_idx, 1]),
                "pred_F3_pressure_cv": float(pred_mean[local_idx, 2]),
                "unc_F1": float(pred_std[local_idx, 0]),
                "unc_F2": float(pred_std[local_idx, 1]),
                "unc_F3": float(pred_std[local_idx, 2]),
                "inj_wells_1based": json.dumps((new_inj[rank] + 1).tolist(), ensure_ascii=False),
                "prod_wells_1based": json.dumps((new_prod[rank] + 1).tolist(), ensure_ascii=False),
            }
        )
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    summary = {
        "base_lhs": str(BASE_LHS),
        "out_lhs": str(OUT_LHS),
        "out_dir": str(OUT_DIR),
        "official_current_model": True,
        "f3_training_mode": "component_log_std_minus_log_mean",
        "start_index": start_idx,
        "stop_index_exclusive": start_idx + 80,
        "n_new": 80,
        "category_counts": counts,
        "quotas": QUOTAS,
        "lhs_candidate_stats": lhs_stats,
        "n_unique_candidates": len(unique_pairs),
        "n_nsga_candidates_loaded": len(nsga_pairs),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved LHS: {OUT_LHS}")
    print(f"Saved selection CSV: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
