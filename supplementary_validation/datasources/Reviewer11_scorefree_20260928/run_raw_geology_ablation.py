"""Train and optionally optimize the no-prescreen-score ablation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_f3_surrogate import ComponentF3Surrogate
from raw_geology_features import RawGeologyFeatureExtractor
from validation_tools.build_static_sideview_fields import read_cmg_txt, read_property_txt


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5),
            "mae": float(mean_absolute_error(y, p))}


def load_fields(args, bound):
    prop = Path(args.property_dir)
    prefix, suffix = args.property_prefix, args.property_suffix
    pressure = read_cmg_txt(prop / f"{prefix}Pressure{suffix}") * 1e-3
    thickness = read_cmg_txt(prop / f"{prefix}Grid Thickness{suffix}")
    poro = read_property_txt(prop / args.porosity_file)
    perm = read_property_txt(prop / args.permeability_file)
    return poro, perm, thickness, pressure, bound


def build_models(Xf, y, std, mean, params):
    models = {k: XGBRegressor(**params) for k in ("F1", "F2", "pressure_std_mpa", "pressure_mean_mpa")}
    models["F1"].fit(Xf, y[:, 0])
    models["F2"].fit(Xf, y[:, 1])
    models["pressure_std_mpa"].fit(Xf, np.log(np.maximum(std, 1e-12)))
    models["pressure_mean_mpa"].fit(Xf, np.log(np.maximum(mean, 1e-12)))
    return models


def predict(models, Xf):
    std = np.exp(models["pressure_std_mpa"].predict(Xf))
    mean = np.exp(models["pressure_mean_mpa"].predict(Xf))
    return np.column_stack((models["F1"].predict(Xf), models["F2"].predict(Xf), std / mean))


def evaluate(y, p):
    return {k: metrics(y[:, i], p[:, i]) for i, k in enumerate(("F1", "F2", "F3"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--property-dir", type=Path, required=True)
    ap.add_argument("--model-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument("--property-prefix", default="2026_0421_1day_property ")
    ap.add_argument("--property-suffix", default=" 2025-Jan-01.txt")
    ap.add_argument("--porosity-file", default="property_260421_all_poro.txt")
    ap.add_argument("--permeability-file", default="property_260421_all_perm.txt")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    with np.load(args.results, allow_pickle=False) as data:
        good = data["success_mask"].astype(bool)
        X = np.asarray(data["X"][good], dtype=float)
        y = np.asarray(data["objectives"][good], dtype=float)
        std = np.asarray(data["pressure_heterogeneity_injend_std_mpa"][good], dtype=float)
        mean = np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][good], dtype=float)
        sample_index = np.asarray(data["sample_index"][good], dtype=int)
    with np.load(args.seeds, allow_pickle=False) as seeds:
        bound = seeds["bound_mask"]
    fields = load_fields(args, bound)
    extractor = RawGeologyFeatureExtractor(*fields)
    Xf = np.hstack((X, extractor(X))).astype(np.float32)
    params = dict(n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.9,
                  colsample_bytree=0.9, reg_lambda=5.0, min_child_weight=3.0,
                  n_jobs=args.n_jobs, tree_method="hist", device="cpu", random_state=42)
    train, test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, shuffle=True)
    split_models = build_models(Xf[train], y[train], std[train], mean[train], params)
    test_pred = predict(split_models, Xf[test])
    train_pred = predict(split_models, Xf[train])
    cv = []
    for cv_train, cv_test in KFold(5, shuffle=True, random_state=42).split(Xf):
        m = build_models(Xf[cv_train], y[cv_train], std[cv_train], mean[cv_train], params)
        cv.append(evaluate(y[cv_test], predict(m, Xf[cv_test])))
    final_models = build_models(Xf, y, std, mean, params)
    model = ComponentF3Surrogate(extractor, final_models, log_components=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    report = {
        "feature_mode": "raw_poro_perm_thickness_pressure",
        "n_raw_features": int(extractor(X[:1]).shape[1]), "n_total_features": int(Xf.shape[1]),
        "n_success": int(len(X)), "split_seed": 42, "test_size": 0.2,
        "feature_names": [f"coordinate_{i + 1}" for i in range(20)] + extractor.feature_names,
        "train_sample_index": sample_index[train].tolist(),
        "test_sample_index": sample_index[test].tolist(),
        "xgb_params": params, "train": evaluate(y[train], train_pred),
        "test": evaluate(y[test], test_pred), "cv": cv,
        "cv_mean": {k: {m: float(np.mean([fold[k][m] for fold in cv])) for m in ("r2", "rmse", "mae")} for k in ("F1", "F2", "F3")},
    }
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
