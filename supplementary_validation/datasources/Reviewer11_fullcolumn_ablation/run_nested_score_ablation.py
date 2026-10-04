"""Matched ablation of the existing 333-feature surrogate.

The only change in the ablation is removal of features derived directly from
the injection/production prescreen score J.  Coordinates, geometry, thickness,
static well-conditioned fields, and rate descriptors are retained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
CODE = Path(r"F:/CODE1")
sys.path.insert(0, str(CODE))

RESULTS = CODE / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
MODEL = CODE / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_model_0_683_static_well_triview_log_20260817.joblib"
METRICS = CODE / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_metrics_0_683_static_well_triview_log_20260817.json"

PARAMS = dict(
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
    random_state=42,
)


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def fit_predict(matrix: np.ndarray, train: np.ndarray, test: np.ndarray,
                y: np.ndarray, std: np.ndarray, mean: np.ndarray) -> np.ndarray:
    targets = (
        y[:, 0],
        y[:, 1],
        np.log(np.maximum(std, 1e-12)),
        np.log(np.maximum(mean, 1e-12)),
    )
    predictions = []
    for target in targets:
        model = XGBRegressor(**PARAMS)
        model.fit(matrix[train], target[train])
        predictions.append(model.predict(matrix[test]))
    return np.column_stack((predictions[0], predictions[1],
                            np.exp(predictions[2] - predictions[3])))


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    return {f"F{i + 1}": metric(y[:, i], pred[:, i]) for i in range(3)}


def summarize(folds: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        target: {
            measure: float(np.mean([fold[target][measure] for fold in folds]))
            for measure in ("r2", "rmse", "mae")
        }
        for target in ("F1", "F2", "F3")
    }


def run_design(matrix: np.ndarray, train: np.ndarray, test: np.ndarray,
               y: np.ndarray, std: np.ndarray, mean: np.ndarray) -> dict:
    holdout = evaluate(y[test], fit_predict(matrix, train, test, y, std, mean))
    folds = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(matrix), 1):
        pred = fit_predict(matrix, tr, te, y, std, mean)
        result = evaluate(y[te], pred)
        folds.append(result)
        print(f"  fold {fold} R2: " + ", ".join(
            f"{name}={result[name]['r2']:.6f}" for name in ("F1", "F2", "F3")
        ), flush=True)
    return {"holdout": holdout, "cv_folds": folds, "cv_mean": summarize(folds)}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with np.load(RESULTS, allow_pickle=False) as data:
        good = data["success_mask"].astype(bool)
        coordinates = np.asarray(data["X"][good], dtype=np.float32)
        y = np.asarray(data["objectives"][good], dtype=np.float64)
        std = np.asarray(data["pressure_heterogeneity_injend_std_mpa"][good], dtype=np.float64)
        mean = np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][good], dtype=np.float64)
        sample_index = np.asarray(data["sample_index"][good], dtype=int)
    assert coordinates.shape == (679, 20)

    official_model = joblib.load(MODEL)
    extractor = official_model.feature_extractor
    names = list(extractor.feature_names)
    assert len(names) == 313 and len(set(names)) == 313
    extracted = np.asarray(extractor(coordinates), dtype=np.float32)
    assert extracted.shape == (679, 313)
    full = np.hstack((coordinates, extracted)).astype(np.float32)
    assert full.shape == (679, 333)

    # J values, J summaries, threshold fractions, and local J-map statistics.
    # Geometry and thickness columns in the same extractor are retained.
    score_names = [
        name for name in names
        if ("J" in name or "top20" in name or "top30" in name)
    ]
    score_names += [
        "inj_J_5x5_mean_mean", "inj_J_5x5_mean_min",
        "inj_J_center_minus_5x5_mean_mean", "inj_J_center_minus_5x5_mean_min",
        "inj_J_grad_mean_mean", "inj_J_grad_max_max",
        "prod_J_5x5_mean_mean", "prod_J_center_minus_5x5_mean_mean",
        "prod_J_grad_mean_mean", "prod_J_grad_max_max",
    ]
    score_names = list(dict.fromkeys(score_names))
    remove_extractor = [i for i, name in enumerate(names) if name in score_names]
    remove_full = [20 + i for i in remove_extractor]
    keep_full = [i for i in range(full.shape[1]) if i not in remove_full]
    scorefree = full[:, keep_full]
    assert len(remove_extractor) == 32
    assert scorefree.shape == (679, 301)
    assert [names[i] for i in remove_extractor] == score_names

    print(f"Full input: {full.shape[1]} features", flush=True)
    print(f"Removed score-derived columns: {len(remove_full)}", flush=True)
    print("Removed names:", flush=True)
    for name in score_names:
        print(f"  - {name}", flush=True)
    print(f"Score-free input: {scorefree.shape[1]} features", flush=True)

    official_metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    train, test = train_test_split(
        np.arange(len(coordinates)), test_size=0.2, random_state=42, shuffle=True
    )
    assert sample_index[train].tolist() == official_metrics["split"]["train_sample_index"]
    assert sample_index[test].tolist() == official_metrics["split"]["test_sample_index"]

    print("Running full 333-feature reproduction", flush=True)
    full_result = run_design(full, train, test, y, std, mean)
    print("Running score-free 301-feature ablation", flush=True)
    scorefree_result = run_design(scorefree, train, test, y, std, mean)

    report = {
        "design": "Nested ablation of the official 333-feature input",
        "n_samples": 679,
        "n_features_full": 333,
        "n_features_scorefree": 301,
        "n_removed_score_features": 32,
        "removed_score_feature_names": score_names,
        "retained_feature_count_including_coordinates": 301,
        "score_definition": "Direct J values, J summaries, top-fraction descriptors, and local J-map statistics; geometry, thickness, static fields, and rates retained.",
        "n_train": len(train),
        "n_test": len(test),
        "split_seed": 42,
        "cv_seed": 42,
        "xgb_params": PARAMS,
        "official_model": str(MODEL),
        "official_metrics": str(METRICS),
        "full_333": full_result,
        "scorefree_301": scorefree_result,
        "delta_scorefree_minus_full": {
            target: {
                measure: scorefree_result["holdout"][target][measure]
                - full_result["holdout"][target][measure]
                for measure in ("r2", "rmse", "mae")
            }
            for target in ("F1", "F2", "F3")
        },
        "train_sample_index": sample_index[train].tolist(),
        "test_sample_index": sample_index[test].tolist(),
    }
    out = HERE / "nested_score_ablation_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RESULT {out}", flush=True)


if __name__ == "__main__":
    main()
