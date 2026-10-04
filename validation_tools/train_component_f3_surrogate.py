from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_f3_surrogate import ComponentF3Surrogate
from validation_tools.train_surrogate import build_feature_extractor


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def _make_model(params):
    return XGBRegressor(**dict(params))


def _fit_models(Xf, y, pressure_std, pressure_mean, params, log_components=False):
    models = {
        "F1": _make_model(params),
        "F2": _make_model(params),
        "pressure_std_mpa": _make_model(params),
        "pressure_mean_mpa": _make_model(params),
    }
    models["F1"].fit(Xf, y[:, 0])
    models["F2"].fit(Xf, y[:, 1])
    if log_components:
        models["pressure_std_mpa"].fit(Xf, np.log(np.maximum(pressure_std, 1e-12)))
        models["pressure_mean_mpa"].fit(Xf, np.log(np.maximum(pressure_mean, 1e-12)))
    else:
        models["pressure_std_mpa"].fit(Xf, pressure_std)
        models["pressure_mean_mpa"].fit(Xf, pressure_mean)
    return models


def _predict_from_models(models, Xf, log_components=False):
    f1 = np.asarray(models["F1"].predict(Xf), dtype=float)
    f2 = np.asarray(models["F2"].predict(Xf), dtype=float)
    std_pred = np.asarray(models["pressure_std_mpa"].predict(Xf), dtype=float)
    mean_pred = np.asarray(models["pressure_mean_mpa"].predict(Xf), dtype=float)
    if log_components:
        std = np.exp(std_pred)
        mean = np.exp(mean_pred)
    else:
        std = np.maximum(std_pred, 0.0)
        mean = np.maximum(mean_pred, 1e-9)
    f3 = std / mean
    return np.column_stack([f1, f2, f3])


def _evaluate(y_true, y_pred):
    return {
        "F1": _metrics(y_true[:, 0], y_pred[:, 0]),
        "F2": _metrics(y_true[:, 1], y_pred[:, 1]),
        "F3": _metrics(y_true[:, 2], y_pred[:, 2]),
    }


def _cv_metrics(Xf, y, pressure_std, pressure_mean, params, splits, seed, log_components=False):
    per_fold = []
    for train_idx, test_idx in KFold(n_splits=splits, shuffle=True, random_state=seed).split(Xf):
        models = _fit_models(
            Xf[train_idx],
            y[train_idx],
            pressure_std[train_idx],
            pressure_mean[train_idx],
            params,
            log_components=log_components,
        )
        pred = _predict_from_models(models, Xf[test_idx], log_components=log_components)
        per_fold.append(_evaluate(y[test_idx], pred))

    out = {}
    for target in ("F1", "F2", "F3"):
        out[target] = {}
        for metric in ("r2", "rmse", "mae", "mape"):
            vals = [fold[target][metric] for fold in per_fold]
            out[target][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the current official surrogate: F1/F2 direct XGBoost, "
            "F3 reconstructed from separately learned pressure std and mean components."
        )
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--feature-mode", default="prescreen_geometry_local")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument(
        "--linear-components",
        action="store_true",
        help="fit pressure std/mean directly; default fits log(std) and log(mean)",
    )
    args = parser.parse_args()
    use_log_components = not bool(args.linear_components)

    data = np.load(args.results, allow_pickle=True)
    success = np.asarray(data["success_mask"], dtype=bool)
    finite = (
        success
        & np.isfinite(data["objectives"]).all(axis=1)
        & np.isfinite(data["pressure_heterogeneity_injend_std_mpa"])
        & np.isfinite(data["pressure_heterogeneity_injend_mean_mpa"])
        & (data["pressure_heterogeneity_injend_mean_mpa"] > 0.0)
    )
    X = np.asarray(data["X"][finite], dtype=float)
    y = np.asarray(data["objectives"][finite], dtype=float)
    pressure_std = np.asarray(data["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float)
    pressure_mean = np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float)

    feature_extractor, feature_report = build_feature_extractor(Path(args.seeds), args.feature_mode)
    Xf = np.hstack([X, np.asarray(feature_extractor(X), dtype=float)]) if feature_extractor else X
    params = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "min_child_weight": float(args.min_child_weight),
        "n_jobs": int(args.n_jobs),
        "tree_method": "hist",
        "device": "cpu",
        "random_state": int(args.split_seed),
    }

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=float(args.test_size),
        random_state=int(args.split_seed),
        shuffle=True,
    )
    split_models = _fit_models(
        Xf[train_idx],
        y[train_idx],
        pressure_std[train_idx],
        pressure_mean[train_idx],
        params,
        log_components=use_log_components,
    )
    test_metrics = _evaluate(
        y[test_idx],
        _predict_from_models(split_models, Xf[test_idx], log_components=use_log_components),
    )
    train_metrics = _evaluate(
        y[train_idx],
        _predict_from_models(split_models, Xf[train_idx], log_components=use_log_components),
    )
    cv_metrics = _cv_metrics(
        Xf,
        y,
        pressure_std,
        pressure_mean,
        params,
        int(args.cv_splits),
        int(args.split_seed),
        log_components=use_log_components,
    )

    final_models = _fit_models(
        Xf,
        y,
        pressure_std,
        pressure_mean,
        params,
        log_components=use_log_components,
    )
    model = ComponentF3Surrogate(
        feature_extractor=feature_extractor,
        models=final_models,
        log_components=use_log_components,
    )
    model_out = Path(args.model_out)
    report_out = Path(args.report_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    report = {
        "results": str(args.results),
        "model_path": str(model_out),
        "n_total_rows": int(len(success)),
        "n_regression_rows": int(len(X)),
        "f3_training_mode": (
            "component_log_std_minus_log_mean"
            if use_log_components
            else "component_std_model_div_mean_model"
        ),
        "official_current_model": bool(use_log_components),
        "notes": (
            "F3 objective is unchanged: pressure std / pressure mean. "
            "Only the surrogate training strategy is changed: std and mean are "
            "learned separately in log space and recombined during prediction."
            if use_log_components
            else "Linear component mode is retained only for ablation."
        ),
        "objective_names": [
            "F1_pressure_rise",
            "F2_plume_area_sg0p1_km2",
            "F3_pressure_CV_std_over_mean",
        ],
        "split": {
            "test_size": float(args.test_size),
            "split_seed": int(args.split_seed),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_sample_index": [int(v) for v in data["sample_index"][finite][train_idx].tolist()],
            "test_sample_index": [int(v) for v in data["sample_index"][finite][test_idx].tolist()],
        },
        "feature_report": feature_report,
        "xgb_params": params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cross_validation": cv_metrics,
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model : {model_out}")
    print(f"Saved report: {report_out}")
    print(json.dumps({"test": test_metrics, "cv": cv_metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
