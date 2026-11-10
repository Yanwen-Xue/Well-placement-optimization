from pathlib import Path
import json
import sys

import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

CODE = Path(r"F:/CODE1")
sys.path.insert(0, str(CODE))
MODEL_DIR = CODE / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683"
sys.path.insert(0, str(MODEL_DIR))

RESULTS = CODE / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
OFFICIAL = MODEL_DIR / "component_f3_model_0_683_static_well_triview_log_20260817.joblib"
OUT = Path(__file__).with_name("grouped_contribution_results.json")

PARAMS = dict(
    n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.9,
    colsample_bytree=0.9, reg_lambda=5.0, min_child_weight=3.0,
    n_jobs=16, tree_method="hist", device="cpu", random_state=42,
)


def fit_models(Xtr, ytr):
    return [
        XGBRegressor(**PARAMS).fit(Xtr, ytr[:, 0]),
        XGBRegressor(**PARAMS).fit(Xtr, ytr[:, 1]),
        XGBRegressor(**PARAMS).fit(Xtr, np.log(np.maximum(ytr[:, 3], 1e-12))),
        XGBRegressor(**PARAMS).fit(Xtr, np.log(np.maximum(ytr[:, 4], 1e-12))),
    ]


def predict(models, Xte):
    p1 = models[0].predict(Xte)
    p2 = models[1].predict(Xte)
    ps = np.exp(models[2].predict(Xte))
    pm = np.exp(models[3].predict(Xte))
    return np.column_stack([p1, p2, ps / pm])


def scores(y, pred):
    return [float(r2_score(y[:, i], pred[:, i])) for i in range(3)]


def main():
    sys.path.insert(0, str(Path(__file__).parent / "Reviewer11_scorefree_20260928"))
    # The serialized official extractor references feature_333; this is the
    # current repository module providing the same LayoutSetFeatureExtractor.
    import surrogate
    sys.modules["feature_333"] = surrogate
    with np.load(RESULTS, allow_pickle=False) as d:
        ok = d["success_mask"].astype(bool)
        Xcoord = np.asarray(d["X"][ok], dtype=float)
        yobj = np.asarray(d["objectives"][ok], dtype=float)
        std = np.asarray(d["pressure_heterogeneity_injend_std_mpa"][ok], dtype=float)
        mean = np.asarray(d["pressure_heterogeneity_injend_mean_mpa"][ok], dtype=float)
    y = np.column_stack([yobj, std, mean])
    base = joblib.load(OFFICIAL).feature_extractor
    Xfull = np.hstack([Xcoord, np.asarray(base(Xcoord), dtype=float)])
    groups = {
        "coordinates": list(range(0, 20)),
        "plan_view_local": list(range(20, 68)),
        "side_view_vertical": list(range(68, 286)),
        "layout_operating": list(range(286, 333)),
    }
    rng = np.random.default_rng(20261003)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    for fold, (tr, te) in enumerate(kf.split(Xfull), 1):
        models = fit_models(Xfull[tr], y[tr])
        pred = predict(models, Xfull[te])
        base_score = scores(y[te, :3], pred)
        for group, cols in groups.items():
            deltas = []
            for rep in range(10):
                perm = rng.permutation(len(te))
                xp = Xfull[te].copy()
                xp[:, cols] = xp[perm][:, cols]
                pp = predict(models, xp)
                deltas.append(np.asarray(base_score) - np.asarray(scores(y[te, :3], pp)))
            deltas = np.asarray(deltas)
            records.append({
                "fold": fold, "group": group, "baseline_r2": base_score,
                "delta_r2_mean": deltas.mean(0).tolist(),
                "delta_r2_sd": deltas.std(0, ddof=1).tolist(),
            })
        print(f"fold {fold} baseline {base_score}", flush=True)
    summary = {}
    for group in groups:
        rs = [r for r in records if r["group"] == group]
        vals = np.asarray([r["delta_r2_mean"] for r in rs])
        summary[group] = {
            "delta_r2_mean": vals.mean(0).tolist(),
            "delta_r2_sd_across_folds": vals.std(0, ddof=1).tolist(),
        }
    result = {"n": int(len(Xfull)), "groups": {k: [int(vv) for vv in v] for k, v in groups.items()}, "summary": summary, "folds": records}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
