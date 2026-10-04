"""Matched XGBoost ablation using unaggregated geological well columns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
CODE = Path("F:/CODE1")
sys.path.insert(0, str(CODE))
from validation_tools.build_static_sideview_fields import read_cmg_txt, read_property_txt

RESULTS = CODE / "_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
OFFICIAL = CODE / "_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_metrics_0_683_static_well_triview_log_20260817.json"
PROPERTY = CODE / "property_CMG"
PARAMS = dict(n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.9,
              colsample_bytree=0.9, reg_lambda=5.0, min_child_weight=3.0,
              n_jobs=16, tree_method="hist", device="cpu", random_state=42)


def score(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred))}


def fit_predict(train, test, matrix, y, std, mean):
    predictions = []
    for target in (y[:, 0], y[:, 1], np.log(np.maximum(std, 1e-12)),
                   np.log(np.maximum(mean, 1e-12))):
        model = XGBRegressor(**PARAMS)
        model.fit(matrix[train], target[train])
        predictions.append(model.predict(matrix[test]))
    result = np.column_stack((predictions[0], predictions[1],
                              np.exp(predictions[2] - predictions[3])))
    return result


def evaluate(y, pred):
    return {f"F{i + 1}": score(y[:, i], pred[:, i]) for i in range(3)}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    with np.load(RESULTS, allow_pickle=False) as data:
        good = data["success_mask"].astype(bool)
        X = np.asarray(data["X"][good], dtype=np.float32)
        y = np.asarray(data["objectives"][good], dtype=np.float64)
        std = np.asarray(data["pressure_heterogeneity_injend_std_mpa"][good], dtype=np.float64)
        mean = np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][good], dtype=np.float64)
        sample_index = np.asarray(data["sample_index"][good], dtype=int)
    assert X.shape == (679, 20)
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    train, test = train_test_split(np.arange(len(X)), test_size=0.2,
                                   random_state=42, shuffle=True)
    assert sample_index[train].tolist() == official["split"]["train_sample_index"]
    assert sample_index[test].tolist() == official["split"]["test_sample_index"]

    prefix, suffix = "2026_0421_1day_property ", " 2025-Jan-01.txt"
    sources = (
        ("porosity", PROPERTY / "property_260421_all_poro.txt", read_property_txt, 1.0),
        ("horizontal_permeability", PROPERTY / "property_260421_all_perm.txt", read_property_txt, 1.0),
        ("cell_thickness", PROPERTY / f"{prefix}Grid Thickness{suffix}", read_cmg_txt, 1.0),
        ("initial_pressure_mpa", PROPERTY / f"{prefix}Pressure{suffix}", read_cmg_txt, 1e-3),
        ("grid_top_depth", PROPERTY / f"{prefix}Grid Top{suffix}", read_cmg_txt, 1.0),
    )
    wells = np.rint(X).astype(int).reshape(-1, 10, 2)
    assert np.all((wells[:, :, 0] >= 0) & (wells[:, :, 0] < 268))
    assert np.all((wells[:, :, 1] >= 0) & (wells[:, :, 1] < 203))
    parts = [X]
    audit = {}
    for name, path, reader, factor in sources:
        field = reader(path)
        assert field.shape == (268, 203, 200)
        columns = (field[wells[:, :, 0], wells[:, :, 1], :] * factor).astype(np.float32)
        assert columns.shape == (679, 10, 200)
        assert np.isfinite(columns).all(), name
        parts.append(columns.reshape(679, -1))
        audit[name] = {"path": str(path), "n_features": int(columns.size // len(X)),
                       "min": float(columns.min()), "max": float(columns.max()),
                       "nonzero_fraction": float(np.mean(columns != 0))}
        print(f"Loaded {name}: {columns.shape}", flush=True)
        del field, columns
    matrix = np.hstack(parts).astype(np.float32)
    assert matrix.shape == (679, 10020)
    print(f"Training matrix {matrix.shape}", flush=True)
    holdout_pred = fit_predict(train, test, matrix, y, std, mean)
    holdout = evaluate(y[test], holdout_pred)
    print("Holdout R2:", {k: v["r2"] for k, v in holdout.items()}, flush=True)
    cv = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(matrix), 1):
        pred = fit_predict(tr, te, matrix, y, std, mean)
        cv.append(evaluate(y[te], pred))
        print(f"Fold {fold} R2:", {k: v["r2"] for k, v in cv[-1].items()}, flush=True)
    summary = {target: {metric: float(np.mean([fold[target][metric] for fold in cv]))
                        for metric in ("r2", "rmse", "mae")}
               for target in ("F1", "F2", "F3")}
    report = {"design": "10 well columns x 200 layers x 5 original geological fields + 20 well coordinates",
              "n_samples": len(X), "n_features": matrix.shape[1], "n_train": len(train),
              "n_test": len(test), "split_seed": 42, "cv_seed": 42,
              "sources": audit, "params": PARAMS,
              "holdout": holdout, "cv_folds": cv, "cv_mean": summary,
              "official_comparison_file": str(OFFICIAL),
              "train_sample_index": sample_index[train].tolist(),
              "test_sample_index": sample_index[test].tolist()}
    out = HERE / "fullcolumn_ablation_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULT", out, flush=True)
    print("CV mean R2:", {k: v["r2"] for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
