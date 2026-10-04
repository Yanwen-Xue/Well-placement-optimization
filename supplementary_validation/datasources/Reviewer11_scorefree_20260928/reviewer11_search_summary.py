"""Summarize score-free geological directions on both searched populations."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "jcp_public_repository"))
from reviewer11_independent_analysis import geometry


def main():
    out = ROOT / "reviewer11_new_analysis"
    with (out / "primitive_search_geology.json").open() as file:
        detail = json.load(file)
    assert detail["same_initial_population"]
    experiments = (
        ("prescreen", ROOT / "ablation_prescreen_nsga3", ROOT / "jcp_333_validation_model.joblib"),
        ("primitive", out / "primitive_nsga3", out / "primitive_full_surrogate.joblib"),
    )
    rows = []
    for name, run, model_path in experiments:
        model = joblib.load(model_path)
        for phase, checkpoint in (("Gen0", "ckpt_gen0000.npz"), ("Gen80", "ckpt_final_population.npz")):
            with np.load(run / checkpoint) as z:
                pop = np.asarray(z["pop"], dtype=int)
            pred = model.predict(pop)
            entry = detail["models"]["raw" if name == "primitive" else "prescreen"]["initial" if phase == "Gen0" else "final"]
            assert entry["n"] == len(pop)
            for target_idx, label in enumerate(("F1", "F2", "F3")):
                selected = np.argsort(pred[:, target_idx])[:len(pop) // 5]
                geo = entry[f"low_{label}_geology_median"]
                rows.append({"model": name, "phase": phase, "branch": label, "n": len(selected),
                             "production_kh_mDm": geo["prod_kh_mDm"],
                             "production_ntg": geo["prod_ntg"],
                             "injection_kh_mDm": geo["inj_kh_mDm"],
                             "injection_longest_sand_m": geo["inj_longest_sand_m"],
                             "injection_pair_spacing_m": geo["inj_pair_distance_m"],
                             "source_sink_centroid_m": float(np.median([geometry(pop[i])[3] for i in selected]))})
    pd.DataFrame(rows).to_csv(out / "search_geological_directions.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
