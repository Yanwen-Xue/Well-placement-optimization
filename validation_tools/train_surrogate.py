\
\
\
\
\
\
\
\
\


from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import surrogate


DEFAULT_RESULTS = ROOT / "opt_runs_lhs_doe" / "results.npz"
DEFAULT_MODEL = ROOT / "opt_runs_lhs_doe" / "surrogate_model.joblib"
DEFAULT_REPORT = ROOT / "opt_runs_lhs_doe" / "surrogate_metrics.json"
DEFAULT_SEEDS = ROOT / "opt_seeds_lhs.npz"


def load_dataset(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(results_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["objectives"], dtype=float)

    if "failure_mask" in data.files:
        failure_mask = np.asarray(data["failure_mask"], dtype=bool)
    else:
        success = np.asarray(data["success_mask"], dtype=bool)
        failure_mask = ~success

    info = {
        "sample_index": np.asarray(data["sample_index"], dtype=int) if "sample_index" in data.files else None,
        "sim_years": np.asarray(data["sim_years"], dtype=int) if "sim_years" in data.files else None,
        "failure_reason": np.asarray(data["failure_reason"]) if "failure_reason" in data.files else None,
        "success_mask": np.asarray(data["success_mask"], dtype=bool),
    }
    return X, y, failure_mask, info


def summarize_dataset(info: dict, failure_mask: np.ndarray) -> dict:
    success_mask = info["success_mask"]
    failure_reason = info["failure_reason"]
    out = {
        "n_total": int(len(success_mask)),
        "n_success": int(success_mask.sum()),
        "n_failure": int(failure_mask.sum()),
    }
    if failure_reason is not None:
        out["failure_reason_counts"] = dict(Counter(str(x) for x in failure_reason.tolist()))
    else:
        out["failure_reason_counts"] = {}
    if info["sim_years"] is not None and len(info["sim_years"]) > 0:
        vals = np.unique(info["sim_years"])
        out["sim_years_unique"] = [int(v) for v in vals.tolist()]
    else:
        out["sim_years_unique"] = []
    return out


def split_indices(
    n_total: int,
    failure_mask: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    idx = np.arange(n_total)
    n_fail = int(failure_mask.sum())
    n_succ = int(n_total - n_fail)
    stratify = failure_mask.astype(int) if n_fail >= 2 and n_succ >= 2 else None
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        shuffle=True,
        stratify=stratify,
    )
    mode = "stratified_by_failure" if stratify is not None else "random"
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), mode


def regression_metrics_or_none(model: surrogate.SurrogateModel, X: np.ndarray, y: np.ndarray) -> dict | None:
    if len(X) == 0:
        return None
    return model.metrics(X, y)


def classifier_metrics_or_none(
    model: surrogate.SurrogateModel,
    X: np.ndarray,
    failure_mask: np.ndarray,
) -> dict:
    out = {
        "trained": bool(model.failure_classifier_fitted),
        "reason": model.failure_reason_,
        "n_total": int(len(failure_mask)),
        "n_failure": int(failure_mask.sum()),
        "n_success": int(len(failure_mask) - failure_mask.sum()),
    }
    if not model.failure_classifier_fitted:
        return out

    y_true = failure_mask.astype(int)
    p_fail = model.predict_failure(X)
    y_pred = (p_fail >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    out.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_failure": float(precision),
            "recall_failure": float(recall),
            "f1_failure": float(f1),
            "auc": (
                float(roc_auc_score(y_true, p_fail))
                if len(np.unique(y_true)) == 2
                else None
            ),
            "threshold": 0.5,
            "confusion_matrix": {
                "tn_success_as_success": int(tn),
                "fp_success_as_failure": int(fp),
                "fn_failure_as_success": int(fn),
                "tp_failure_as_failure": int(tp),
            },
        }
    )
    return out


def build_feature_extractor(seeds_path: Path, feature_mode: str):
    if feature_mode == "coords":
        return None, {
            "enabled": False,
            "mode": feature_mode,
            "seeds_path": None,
            "feature_names": [],
        }
    if feature_mode == "prescreen_geometry_local_triview":
        seeds = np.load(seeds_path, allow_pickle=False)
        required = ["J_inj_2d", "J_prod_2d", "bound_mask", "J_conn_xy", "J_conn_xz", "J_conn_yz"]
        missing = [k for k in required if k not in seeds.files]
        if missing:
            raise KeyError(f"missing tri-view prescreen arrays in {seeds_path}: {missing}")
        base = surrogate.make_prescreen_extractor(
            seeds["J_inj_2d"],
            seeds["J_prod_2d"],
            seeds["bound_mask"],
            grid_top_2d=seeds["grid_top_2d"] if "grid_top_2d" in seeds.files else None,
            grid_bot_2d=seeds["grid_bot_2d"] if "grid_bot_2d" in seeds.files else None,
            include_geometry=True,
            include_local=True,
        )
        triview = surrogate.make_triview_extractor(
            seeds["J_conn_xy"],
            seeds["J_conn_xz"],
            seeds["J_conn_yz"],
            inj_k_range=(
                int(seeds["INJ_K_START"]) if "INJ_K_START" in seeds.files else 41,
                int(seeds["INJ_K_END"]) if "INJ_K_END" in seeds.files else 80,
            ),
            prod_k_range=(
                int(seeds["PRD_K_START"]) if "PRD_K_START" in seeds.files else 1,
                int(seeds["PRD_K_END"]) if "PRD_K_END" in seeds.files else 15,
            ),
            bridge_k_range=(
                int(seeds["PRD_K_END"]) + 1 if "PRD_K_END" in seeds.files else 16,
                int(seeds["INJ_K_START"]) - 1 if "INJ_K_START" in seeds.files else 40,
            ),
        )
        layout = surrogate.LayoutSetFeatureExtractor(
            ni=int(seeds["NI"]) if "NI" in seeds.files else seeds["bound_mask"].shape[0],
            nj=int(seeds["NJ"]) if "NJ" in seeds.files else seeds["bound_mask"].shape[1],
            dx=float(seeds["dx"]) if "dx" in seeds.files else 50.0,
        )
        extractor = surrogate.CombinedFeatureExtractor([base, triview, layout])
        feature_names = list(getattr(extractor, "feature_names", []))
        return extractor, {
            "enabled": True,
            "mode": feature_mode,
            "seeds_path": str(seeds_path),
            "feature_names": feature_names,
            "n_prescreen_features": int(len(feature_names)),
            "triview_enabled": True,
            "layout_set_enabled": True,
        }
    if feature_mode == "prescreen_geometry_local_well_triview":
        seeds = np.load(seeds_path, allow_pickle=False)
        required = [
            "J_inj_2d",
            "J_prod_2d",
            "bound_mask",
            "J_inj_3d",
            "J_prod_3d",
        ]
        missing = [k for k in required if k not in seeds.files]
        if missing:
            raise KeyError(f"missing well-conditioned tri-view arrays in {seeds_path}: {missing}")
        base = surrogate.make_prescreen_extractor(
            seeds["J_inj_2d"],
            seeds["J_prod_2d"],
            seeds["bound_mask"],
            grid_top_2d=seeds["grid_top_2d"] if "grid_top_2d" in seeds.files else None,
            grid_bot_2d=seeds["grid_bot_2d"] if "grid_bot_2d" in seeds.files else None,
            include_geometry=True,
            include_local=True,
        )
        well_triview = surrogate.WellConditionedTriViewFeatureExtractor(
            seeds["J_inj_3d"],
            seeds["J_prod_3d"],
            bound_mask=seeds["bound_mask"],
            J_global_max_inj=seeds["J_global_max_inj"] if "J_global_max_inj" in seeds.files else None,
            J_global_max_prod=seeds["J_global_max_prod"] if "J_global_max_prod" in seeds.files else None,
            inj_k_range=(
                int(seeds["INJ_K_START"]) if "INJ_K_START" in seeds.files else 41,
                int(seeds["INJ_K_END"]) if "INJ_K_END" in seeds.files else 80,
            ),
            prod_k_range=(
                int(seeds["PRD_K_START"]) if "PRD_K_START" in seeds.files else 1,
                int(seeds["PRD_K_END"]) if "PRD_K_END" in seeds.files else 15,
            ),
            bridge_k_range=(
                int(seeds["PRD_K_END"]) + 1 if "PRD_K_END" in seeds.files else 16,
                int(seeds["INJ_K_START"]) - 1 if "INJ_K_START" in seeds.files else 40,
            ),
        )
        layout = surrogate.LayoutSetFeatureExtractor(
            ni=int(seeds["NI"]) if "NI" in seeds.files else seeds["bound_mask"].shape[0],
            nj=int(seeds["NJ"]) if "NJ" in seeds.files else seeds["bound_mask"].shape[1],
            dx=float(seeds["dx"]) if "dx" in seeds.files else 50.0,
        )
        extractor = surrogate.CombinedFeatureExtractor([base, well_triview, layout])
        feature_names = list(getattr(extractor, "feature_names", []))
        return extractor, {
            "enabled": True,
            "mode": feature_mode,
            "seeds_path": str(seeds_path),
            "feature_names": feature_names,
            "n_prescreen_features": int(len(feature_names)),
            "well_conditioned_triview_enabled": True,
            "layout_set_enabled": True,
        }
    if feature_mode == "prescreen_geometry_local_static_well_triview":
        seeds = np.load(seeds_path, allow_pickle=False)
        static_path = seeds_path.parent / "prescreen_maps" / "static_sideview_fields_no_pressure_threshold.npz"
        required = [
            "J_inj_2d",
            "J_prod_2d",
            "bound_mask",
        ]
        missing = [k for k in required if k not in seeds.files]
        if missing:
            raise KeyError(f"missing official XY prescreen arrays in {seeds_path}: {missing}")
        if not static_path.exists():
            raise FileNotFoundError(
                f"static side-view fields not found: {static_path}. "
                "Run validation_tools/build_static_sideview_fields.py first."
            )
        static = np.load(static_path, allow_pickle=False)
        static_required = ["static_inj_3d", "static_prod_3d", "pressure_initial_3d"]
        static_missing = [k for k in static_required if k not in static.files]
        if static_missing:
            raise KeyError(f"missing static side-view arrays in {static_path}: {static_missing}")
        base = surrogate.make_prescreen_extractor(
            seeds["J_inj_2d"],
            seeds["J_prod_2d"],
            seeds["bound_mask"],
            grid_top_2d=seeds["grid_top_2d"] if "grid_top_2d" in seeds.files else None,
            grid_bot_2d=seeds["grid_bot_2d"] if "grid_bot_2d" in seeds.files else None,
            include_geometry=True,
            include_local=True,
        )
        static_triview = surrogate.StaticWellConditionedTriViewFeatureExtractor(
            static["static_inj_3d"],
            static["static_prod_3d"],
            static["pressure_initial_3d"],
            inj_k_range=(
                int(static["INJ_K_START"]) if "INJ_K_START" in static.files else 41,
                int(static["INJ_K_END"]) if "INJ_K_END" in static.files else 80,
            ),
            prod_k_range=(
                int(static["PRD_K_START"]) if "PRD_K_START" in static.files else 1,
                int(static["PRD_K_END"]) if "PRD_K_END" in static.files else 15,
            ),
            bridge_k_range=(
                int(static["PRD_K_END"]) + 1 if "PRD_K_END" in static.files else 16,
                int(static["INJ_K_START"]) - 1 if "INJ_K_START" in static.files else 40,
            ),
        )
        layout = surrogate.LayoutSetFeatureExtractor(
            ni=int(seeds["NI"]) if "NI" in seeds.files else seeds["bound_mask"].shape[0],
            nj=int(seeds["NJ"]) if "NJ" in seeds.files else seeds["bound_mask"].shape[1],
            dx=float(seeds["dx"]) if "dx" in seeds.files else 50.0,
        )
        extractor = surrogate.CombinedFeatureExtractor([base, static_triview, layout])
        feature_names = list(getattr(extractor, "feature_names", []))
        return extractor, {
            "enabled": True,
            "mode": feature_mode,
            "seeds_path": str(seeds_path),
            "static_sideview_path": str(static_path),
            "feature_names": feature_names,
            "n_prescreen_features": int(len(feature_names)),
            "static_sideview_without_pressure_threshold": True,
            "layout_set_enabled": True,
        }
    seeds = np.load(seeds_path, allow_pickle=False)
    required = ["J_inj_2d", "J_prod_2d", "bound_mask"]
    missing = [k for k in required if k not in seeds.files]
    if missing:
        raise KeyError(f"missing prescreen arrays in {seeds_path}: {missing}")
    include_geometry = feature_mode in {"prescreen_geometry", "prescreen_geometry_local"}
    include_local = feature_mode == "prescreen_geometry_local"
    extractor = surrogate.make_prescreen_extractor(
        seeds["J_inj_2d"],
        seeds["J_prod_2d"],
        seeds["bound_mask"],
        grid_top_2d=seeds["grid_top_2d"] if "grid_top_2d" in seeds.files else None,
        grid_bot_2d=seeds["grid_bot_2d"] if "grid_bot_2d" in seeds.files else None,
        include_geometry=include_geometry,
        include_local=include_local,
    )
    feature_names = list(getattr(extractor, "feature_names", []))
    return extractor, {
        "enabled": True,
        "mode": feature_mode,
        "seeds_path": str(seeds_path),
        "feature_names": feature_names,
        "n_prescreen_features": int(len(feature_names)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CO2 well-placement surrogate.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="path to DOE results.npz")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL), help="joblib output path")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT), help="metrics report path")
    parser.add_argument("--cv-splits", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.20, help="held-out test fraction")
    parser.add_argument("--split-seed", type=int, default=42, help="train/test split seed")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="boosting rounds without validation improvement before stopping; use 0 to disable")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="fraction of the training fold used for early stopping validation")
    parser.add_argument("--n-jobs", type=int, default=128, help="XGBoost CPU threads per model")
    parser.add_argument("--n-estimators", type=int, default=300, help="XGBoost boosting rounds before early stopping")
    parser.add_argument("--max-depth", type=int, default=5, help="XGBoost maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--subsample", type=float, default=0.85, help="XGBoost row subsampling ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.85, help="XGBoost column subsampling ratio")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="XGBoost L2 regularization")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="XGBoost L1 regularization")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="XGBoost minimum child weight")
    parser.add_argument(
        "--target-params-json",
        default=None,
        help=(
            "optional JSON file with per-target XGBoost overrides, e.g. "
            '{"F1":{"max_depth":2},"F2":{"reg_lambda":3},"F3":{"subsample":0.8}}'
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="XGBoost training device; use cuda for NVIDIA GPU acceleration",
    )
    parser.add_argument(
        "--tree-method",
        default="hist",
        choices=["hist", "approx", "exact"],
        help="XGBoost tree construction method",
    )
    parser.add_argument("--ref-mode", default="medoid", choices=["first", "lex", "medoid"])
    parser.add_argument("--with-uncertainty", action="store_true", help="train quantile heads")
    parser.add_argument("--no-failure-classifier", action="store_true", help="disable failure classifier head")
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS), help="path to opt_seeds*.npz containing prescreen maps")
    parser.add_argument("--no-prescreen-features", action="store_true", help="disable prescreen-map feature augmentation")
    parser.add_argument(
        "--feature-mode",
        choices=[
            "coords",
            "prescreen",
            "prescreen_geometry",
            "prescreen_geometry_local",
            "prescreen_geometry_local_triview",
            "prescreen_geometry_local_well_triview",
            "prescreen_geometry_local_static_well_triview",
        ],
        default="prescreen",
        help="surrogate input feature set; --no-prescreen-features is an alias for coords",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    seeds_path = Path(args.seeds)
    model_out = Path(args.model_out)
    report_out = Path(args.report_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    X, y, failure_mask, info = load_dataset(results_path)
    dataset_summary = summarize_dataset(info, failure_mask)
    if dataset_summary["n_success"] < 5:
        raise ValueError("too few successful samples for surrogate regression")
    if dataset_summary["sim_years_unique"] and dataset_summary["sim_years_unique"] != [80]:
        print(
            "WARNING: surrogate dataset is not pure 80-year data:",
            dataset_summary["sim_years_unique"],
        )
    if not (0.0 < float(args.test_size) < 1.0):
        raise ValueError("--test-size must be between 0 and 1")
    if not (0.0 < float(args.validation_fraction) < 0.5):
        raise ValueError("--validation-fraction must be between 0 and 0.5")

    xgb_params = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "min_child_weight": float(args.min_child_weight),
        "n_jobs": int(args.n_jobs),
        "tree_method": args.tree_method,
        "device": args.device,
    }
    xgb_classifier_params = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "min_child_weight": float(args.min_child_weight),
        "n_jobs": int(args.n_jobs),
        "tree_method": args.tree_method,
        "device": args.device,
    }
    xgb_target_params = None
    if args.target_params_json:
        with open(args.target_params_json, "r", encoding="utf-8") as f:
            xgb_target_params = json.load(f)
    early_stopping_rounds = (
        int(args.early_stopping_rounds)
        if int(args.early_stopping_rounds) > 0
        else None
    )

    train_idx, test_idx, split_mode = split_indices(
        len(X),
        failure_mask,
        test_size=float(args.test_size),
        seed=int(args.split_seed),
    )
    train_success = ~failure_mask[train_idx]
    test_success = ~failure_mask[test_idx]
    feature_mode = "coords" if bool(args.no_prescreen_features) else args.feature_mode
    feature_extractor, feature_report = build_feature_extractor(seeds_path, feature_mode)

    model = surrogate.SurrogateModel(
        feature_extractor=feature_extractor,
        with_uncertainty=bool(args.with_uncertainty),
        with_failure_classifier=not bool(args.no_failure_classifier),
        ref_mode=args.ref_mode,
        xgb_params=xgb_params,
        xgb_target_params=xgb_target_params,
        xgb_classifier_params=xgb_classifier_params,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=float(args.validation_fraction),
    )
    cv = model.cross_validate(
        X[train_idx],
        y[train_idx],
        failure_mask= failure_mask[train_idx],
        n_splits=int(args.cv_splits),
        seed=int(args.split_seed),
    )
    model.fit(X[train_idx], y[train_idx], failure_mask=failure_mask[train_idx])
    train_metrics = regression_metrics_or_none(
        model,
        X[train_idx][train_success],
        y[train_idx][train_success],
    )
    test_metrics = regression_metrics_or_none(
        model,
        X[test_idx][test_success],
        y[test_idx][test_success],
    )
    train_classifier_metrics = classifier_metrics_or_none(
        model,
        X[train_idx],
        failure_mask[train_idx],
    )
    test_classifier_metrics = classifier_metrics_or_none(
        model,
        X[test_idx],
        failure_mask[test_idx],
    )
    model.save(model_out)

    report = {
        "dataset": dataset_summary,
        "split": {
            "mode": split_mode,
            "test_size": float(args.test_size),
            "split_seed": int(args.split_seed),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_success": int(train_success.sum()),
            "n_train_failure": int((~train_success).sum()),
            "n_test_success": int(test_success.sum()),
            "n_test_failure": int((~test_success).sum()),
            "train_sample_index": (
                [int(v) for v in info["sample_index"][train_idx].tolist()]
                if info["sample_index"] is not None
                else [int(v) for v in train_idx.tolist()]
            ),
            "test_sample_index": (
                [int(v) for v in info["sample_index"][test_idx].tolist()]
                if info["sample_index"] is not None
                else [int(v) for v in test_idx.tolist()]
            ),
        },
        "model_path": str(model_out),
        "with_uncertainty": bool(args.with_uncertainty),
        "with_failure_classifier": not bool(args.no_failure_classifier),
        "prescreen_features": feature_report,
        "ref_mode": args.ref_mode,
        "xgb_params": xgb_params,
        "xgb_target_params": xgb_target_params,
        "xgb_params_by_target": model.xgb_params_by_target,
        "xgb_classifier_params": xgb_classifier_params,
        "early_stopping_rounds": early_stopping_rounds,
        "validation_fraction": float(args.validation_fraction),
        "best_iterations": model.best_iterations_,
        "cross_validation_train_only": cv,
        "train_metrics_success_only": train_metrics,
        "test_metrics_success_only": test_metrics,
        "train_classifier_metrics": train_classifier_metrics,
        "test_classifier_metrics": test_classifier_metrics,
        "classifier_trained": bool(model.failure_classifier_fitted),
        "classifier_reason": model.failure_reason_,
        "n_features": int(model.n_features),
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved surrogate model : {model_out}")
    print(f"Saved metrics report  : {report_out}")
    print(
        f"Dataset summary       : success={dataset_summary['n_success']} "
        f"failure={dataset_summary['n_failure']}"
    )
    print(
        f"Train/test split      : train={len(train_idx)} "
        f"test={len(test_idx)} mode={split_mode}"
    )
    print(
        f"XGBoost training      : n_jobs={int(args.n_jobs)} "
        f"device={args.device} tree_method={args.tree_method} "
        f"early_stopping_rounds={early_stopping_rounds}"
    )
    print(
        "Prescreen features    : "
        f"{'enabled' if feature_report['enabled'] else 'disabled'} "
        f"n={feature_report.get('n_prescreen_features', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
