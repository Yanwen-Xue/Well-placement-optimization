"""Compare prescreen and separate primitive-field inputs at matched sample budgets."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "jcp_public_repository"))
from component_f3_surrogate import ComponentF3Surrogate
from raw_geology_features import RawGeologyFeatureExtractor
from validation_tools.build_static_sideview_fields import read_cmg_txt, read_property_txt
from validation_tools.run_raw_geology_ablation import build_models, evaluate, predict
from reviewer11_primitive_features import PrimitiveFieldExtractor

OUT = ROOT / "reviewer11_new_analysis"
PROPERTY = Path("F:/CODE1/property_CMG")
RESULTS = ROOT / "jcp_public_repository/public_data/training_responses_684.npz"
SEEDS = ROOT / "jcp_optimizer_seeds_validation.npz"
PARAMS = dict(n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.9,
              colsample_bytree=0.9, reg_lambda=5.0, min_child_weight=3.0,
              n_jobs=16, tree_method="hist", device="cpu", random_state=42)


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(RESULTS) as data:
        good = data["success_mask"].astype(bool)
        X = np.asarray(data["X"][good], dtype=float)
        y = np.asarray(data["objectives"][good], dtype=float)
        std = np.asarray(data["pressure_heterogeneity_injend_std_mpa"][good], dtype=float)
        mean = np.asarray(data["pressure_heterogeneity_injend_mean_mpa"][good], dtype=float)
        ids = data["sample_index"][good]
    with np.load(SEEDS) as data:
        bound = data["bound_mask"]
    prefix, suffix = "2026_0421_1day_property ", " 2025-Jan-01.txt"
    base = RawGeologyFeatureExtractor(
        read_property_txt(PROPERTY / "property_260421_all_poro.txt"),
        read_property_txt(PROPERTY / "property_260421_all_perm.txt"),
        read_cmg_txt(PROPERTY / f"{prefix}Grid Thickness{suffix}"),
        read_cmg_txt(PROPERTY / f"{prefix}Pressure{suffix}") * 1e-3,
        bound,
    )
    raw = PrimitiveFieldExtractor(base, read_cmg_txt(PROPERTY / f"{prefix}Grid Top{suffix}"))
    original = joblib.load(ROOT / "jcp_333_validation_model.joblib")
    features = {
        "prescreen": np.hstack((X, original.feature_extractor(X))).astype(np.float32),
        "primitive": np.hstack((X, raw(X))).astype(np.float32),
    }
    assert features["prescreen"].shape == (679, 333)
    assert features["primitive"].shape == (679, 363)
    train, test = train_test_split(np.arange(len(X)), test_size=.2, random_state=42, shuffle=True)
    official = json.loads((ROOT / "jcp_333_validation_metrics.json").read_text())
    assert ids[train].tolist() == official["split"]["train_sample_index"]
    assert ids[test].tolist() == official["split"]["test_sample_index"]

    report = {"n": len(X), "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist(),
              "feature_counts": {k: v.shape[1] for k, v in features.items()},
              "params": PARAMS, "holdout": {}, "cv": {}, "learning_curve": []}
    for label, matrix in features.items():
        model = build_models(matrix[train], y[train], std[train], mean[train], PARAMS)
        report["holdout"][label] = evaluate(y[test], predict(model, matrix[test]))
        folds = []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(matrix):
            fitted = build_models(matrix[tr], y[tr], std[tr], mean[tr], PARAMS)
            folds.append(evaluate(y[te], predict(fitted, matrix[te])))
        report["cv"][label] = {target: float(np.mean([f[target]["r2"] for f in folds])) for target in ("F1", "F2", "F3")}
        if label == "primitive":
            final = build_models(matrix, y, std, mean, PARAMS)
            joblib.dump(ComponentF3Surrogate(raw, final, log_components=True), OUT / "primitive_full_surrogate.joblib")
    for fraction in (.25, .5, 1.0):
        repeats = (42, 43, 44) if fraction < 1.0 else (42,)
        for seed in repeats:
            chosen = np.random.default_rng(seed).permutation(train)[:round(len(train) * fraction)]
            subset = train[np.isin(train, chosen)]
            for label, matrix in features.items():
                model = build_models(matrix[subset], y[subset], std[subset], mean[subset], PARAMS)
                scores = evaluate(y[test], predict(model, matrix[test]))
                for target in ("F1", "F2", "F3"):
                    report["learning_curve"].append({"model": label, "train_fraction": fraction,
                                                      "train_n": len(subset), "seed": seed,
                                                      "target": target, "holdout_r2": scores[target]["r2"]})
    (OUT / "primitive_input_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, target in zip(axes, ("F1", "F2", "F3")):
        for label, color in (("prescreen", "#23756d"), ("primitive", "#c76945")):
            rows = [r for r in report["learning_curve"] if r["model"] == label and r["target"] == target]
            xs = sorted({r["train_n"] for r in rows})
            ys = [np.mean([r["holdout_r2"] for r in rows if r["train_n"] == n]) for n in xs]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.set(xlabel="Training layouts", ylabel="Holdout R2", title=target)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "primitive_learning_curve.png", dpi=220)
    plt.close(fig)
    print("Holdout:", {k: {t: round(v[t]["r2"], 3) for t in ("F1", "F2", "F3")} for k, v in report["holdout"].items()})
    print("CV:", report["cv"])


if __name__ == "__main__":
    main()
