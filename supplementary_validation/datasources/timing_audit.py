from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

CODE1 = Path(r"F:/CODE1")
sys.path.insert(0, str(CODE1))

import component_f3_surrogate  # noqa: F401
import surrogate  # noqa: F401

RESULTS = CODE1 / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
MODEL = CODE1 / r"_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_model_0_683_static_well_triview_log_20260817.joblib"
SEEDS = CODE1 / r"opt_seeds_0_683_current_mixed_active60_20260813.npz"
OUT = Path(__file__).with_name("timing_audit_333dim.json")


def stamp():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    w0, c0 = start
    return {"wall_s": time.perf_counter() - w0, "cpu_s": time.process_time() - c0}


def main():
    data = np.load(RESULTS, allow_pickle=True)
    success = np.asarray(data["success_mask"], dtype=bool)
    finite = success & np.isfinite(data["objectives"]).all(axis=1)
    X = np.asarray(data["X"][finite], dtype=float)
    y = np.asarray(data["objectives"][finite], dtype=float)
    # The optimizer used 684 seeds; the training set contains the 679 valid layouts.
    model = joblib.load(MODEL)
    extractor = model.feature_extractor
    rng = np.random.default_rng(42)
    base = np.asarray(X, dtype=float)
    # Use the exact optimizer population size and repeat valid layouts only to isolate
    # feature-construction cost from candidate-generation cost.
    opt_ckpt = Path(r"writen/datasources/R1_9_smoothing_audit/search_sigma_6/ckpt_gen0080.npz")
    if opt_ckpt.exists():
        pop = np.asarray(np.load(opt_ckpt, allow_pickle=False)["pop"], dtype=float)
    else:
        pop = np.asarray(base[:240], dtype=float)
    candidates = np.tile(pop, (80, 1))[: 80 * len(pop)]

    timings = {"n_training_layouts": int(len(base)), "n_search_evaluations": int(len(candidates)), "n_features": 333}

    s = stamp()
    train_features = np.asarray(extractor(base), dtype=float)
    timings["feature_construction_training"] = elapsed(s)
    timings["feature_construction_training"]["shape"] = list(train_features.shape)

    s = stamp()
    search_features = np.asarray(extractor(candidates), dtype=float)
    timings["feature_construction_search"] = elapsed(s)
    timings["feature_construction_search"]["shape"] = list(search_features.shape)

    Xf = np.hstack([base, train_features])
    # Fit the same four regressors and parameters used by the official surrogate.
    params = model.models["F1"].get_params()
    fit_times = {}
    targets = {
        "F1": y[:, 0],
        "F2": y[:, 1],
        "pressure_std_mpa": np.asarray(data["pressure_heterogeneity_injend_std_mpa"][finite], dtype=float),
        "pressure_mean_mpa": np.log(np.maximum(np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][finite], dtype=float), 1e-12)),
    }
    # The official model fits log(std) and log(mean) for the pressure components.
    targets["pressure_std_mpa"] = np.log(np.maximum(targets["pressure_std_mpa"], 1e-12))
    for name, target in targets.items():
        estimator = clone(model.models[name])
        s = stamp()
        estimator.fit(Xf, target)
        fit_times[name] = elapsed(s)
    timings["xgboost_four_regressors"] = fit_times
    timings["xgboost_four_regressors_total"] = {
        "wall_s": float(sum(v["wall_s"] for v in fit_times.values())),
        "cpu_s": float(sum(v["cpu_s"] for v in fit_times.values())),
    }

    # Aggregate the run timing from the official optimizer log. The logged run has
    # 240 initial candidates plus 80 generations x 240 offspring evaluations.
    timings["optimizer_log"] = {
        "initial_population": 240,
        "offspring_evaluations": 80 * 240,
        "total_surrogate_evaluations": 240 + 80 * 240,
        "evolution_wall_s": 12 * 60 + 26.465,
        "evolution_wall_min": (12 * 60 + 26.465) / 60.0,
        "source": "F:/CODE1/_organized_active_work_20260817/05_nsga3_current_0_683_20260817/optimizer.log",
    }
    OUT.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
