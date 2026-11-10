from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

CODE = Path(r"F:/CODE1")
sys.path.insert(0, str(CODE))

RESULTS = CODE / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
MODEL = CODE / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_model_0_683_static_well_triview_log_20260817.joblib"
METRICS = CODE / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_metrics_0_683_static_well_triview_log_20260817.json"
OUT = Path(__file__).with_name("planview_sideview_ablation_results.json")

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


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def fit_predict(matrix: np.ndarray, train: np.ndarray, test: np.ndarray,
                y: np.ndarray, pressure_std: np.ndarray,
                pressure_mean: np.ndarray) -> np.ndarray:
    targets = (
        y[:, 0], y[:, 1],
        np.log(np.maximum(pressure_std, 1e-12)),
        np.log(np.maximum(pressure_mean, 1e-12)),
    )
    predictions = []
    for target in targets:
        model = XGBRegressor(**PARAMS)
        model.fit(matrix[train], target[train])
        predictions.append(model.predict(matrix[test]))
    return np.column_stack((predictions[0], predictions[1],
                            np.exp(predictions[2] - predictions[3])))


def evaluate(matrix: np.ndarray, train: np.ndarray, test: np.ndarray,
             y: np.ndarray, pressure_std: np.ndarray,
             pressure_mean: np.ndarray) -> dict:
    prediction = fit_predict(matrix, train, test, y, pressure_std, pressure_mean)
    return {f"F{i + 1}": score(y[test, i], prediction[:, i]) for i in range(3)}


def main() -> None:
    with np.load(RESULTS, allow_pickle=False) as data:
        successful = data["success_mask"].astype(bool)
        coordinates = np.asarray(data["X"][successful], dtype=np.float32)
        objectives = np.asarray(data["objectives"][successful], dtype=np.float64)
        pressure_std = np.asarray(
            data["pressure_heterogeneity_injend_std_mpa"][successful], dtype=np.float64
        )
        pressure_mean = np.asarray(
            data["pressure_heterogeneity_injend_mean_mpa"][successful], dtype=np.float64
        )
        sample_index = np.asarray(data["sample_index"][successful], dtype=int)

    extractor = joblib.load(MODEL).feature_extractor
    names = list(extractor.feature_names)
    extracted = np.asarray(extractor(coordinates), dtype=np.float32)
    assert extracted.shape == (679, len(names))

    score_indices = [
        i for i, name in enumerate(names)
        if name in {f"{well}_J_{stat}" for well in ("inj", "prod")
                    for stat in (*map(str, range(1, 6)), "mean", "min", "max", "std")}
        or name in {f"{well}_top{fraction}_frac" for well in ("inj", "prod")
                    for fraction in (20, 30)}
    ]
    sideview_indices = [
        i for i, name in enumerate(names)
        if name.startswith(("injw_static_", "prodw_static_", "inj_prod_static_", "static_inj_"))
    ]
    assert len(score_indices) == 22
    assert len(sideview_indices) == 218

    planview = np.hstack((coordinates, extracted[:, score_indices])).astype(np.float32)
    planview_sideview = np.hstack(
        (coordinates, extracted[:, score_indices], extracted[:, sideview_indices])
    ).astype(np.float32)

    official = json.loads(METRICS.read_text(encoding="utf-8"))
    train, test = train_test_split(
        np.arange(len(coordinates)), test_size=0.2, random_state=42, shuffle=True
    )
    assert sample_index[train].tolist() == official["split"]["train_sample_index"]
    assert sample_index[test].tolist() == official["split"]["test_sample_index"]

    results = {}
    for label, matrix in (("planview_42", planview),
                          ("planview_plus_sideview_260", planview_sideview)):
        folds = []
        for fold, (train_fold, test_fold) in enumerate(
            KFold(5, shuffle=True, random_state=42).split(matrix), 1
        ):
            fold_result = evaluate(
                matrix, train_fold, test_fold, objectives, pressure_std, pressure_mean
            )
            folds.append(fold_result)
            print(label, "fold", fold,
                  [round(fold_result[f"F{i}"]["r2"], 6) for i in range(1, 4)],
                  flush=True)
        results[label] = {
            "n_features": int(matrix.shape[1]),
            "cv_mean_r2": {
                f"F{i}": float(np.mean([fold[f"F{i}"]["r2"] for fold in folds]))
                for i in range(1, 4)
            },
            "cv_std_r2": {
                f"F{i}": float(np.std([fold[f"F{i}"]["r2"] for fold in folds], ddof=0))
                for i in range(1, 4)
            },
            "holdout": evaluate(
                matrix, train, test, objectives, pressure_std, pressure_mean
            ),
        }

    report = {
        "design": "Matched plan-view versus plan-view plus side-view feature ablation",
        "n_samples": int(len(coordinates)),
        "n_train": int(len(train)),
        "n_holdout": int(len(test)),
        "cv_folds": 5,
        "split_seed": 42,
        "xgboost_parameters": PARAMS,
        "planview_features": [names[i] for i in score_indices],
        "sideview_features": [names[i] for i in sideview_indices],
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RESULT {OUT}", flush=True)


if __name__ == "__main__":
    main()
