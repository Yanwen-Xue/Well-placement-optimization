"""Compare raw geology sampled by prescreen and direct-variable searches."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_tools.build_static_sideview_fields import read_cmg_txt, read_property_txt


def layout_metrics(pop, perm, poro, thick, sand):
    metrics = []
    for row in np.asarray(pop, dtype=int):
        record = {}
        groups = {"inj": (row[:10].reshape(5, 2), slice(40, 80)),
                  "prod": (row[10:].reshape(5, 2), slice(0, 15))}
        for group, (wells, layers) in groups.items():
            k = perm[wells[:, 0], wells[:, 1], layers]
            phi = poro[wells[:, 0], wells[:, 1], layers]
            h = thick[wells[:, 0], wells[:, 1], layers]
            record[f"{group}_kh_mDm"] = float(np.mean(np.sum(k * h, axis=1)))
            record[f"{group}_phi_weighted"] = float(np.mean(np.sum(phi * h, axis=1) / np.maximum(h.sum(axis=1), 1e-9)))
            facies = sand[layers][:, wells[:, 0], wells[:, 1]].T
            record[f"{group}_ntg"] = float(np.mean(np.sum(h * facies, axis=1) / np.maximum(h.sum(axis=1), 1e-9)))
            record[f"{group}_net_sand_m"] = float(np.mean(np.sum(h * facies, axis=1)))
            record[f"{group}_high_k_fraction"] = float(np.mean(np.sum(h * (k >= 4.0), axis=1) / np.maximum(h.sum(axis=1), 1e-9)))
            low = k < 4.0
            runs = []
            sand_runs = []
            for low_column, sand_column, heights in zip(low, facies, h):
                longest = current = 0.0
                sand_longest = sand_current = 0.0
                for is_low, is_sand, height in zip(low_column, sand_column, heights):
                    current = current + height if is_low else 0.0
                    longest = max(longest, current)
                    sand_current = sand_current + height if is_sand else 0.0
                    sand_longest = max(sand_longest, sand_current)
                runs.append(longest)
                sand_runs.append(sand_longest)
            record[f"{group}_longest_low_k_m"] = float(np.mean(runs))
            record[f"{group}_longest_sand_m"] = float(np.mean(sand_runs))
        inj = groups["inj"][0].astype(float) * 50.0
        record["inj_pair_distance_m"] = float(np.mean([np.linalg.norm(inj[i] - inj[j]) for i in range(5) for j in range(i + 1, 5)]))
        metrics.append(record)
    return {key: np.asarray([row[key] for row in metrics]) for key in metrics[0]}


def summarize(values, indices):
    return {key: float(np.median(x[indices])) for key, x in values.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prescreen-run", type=Path, required=True)
    ap.add_argument("--raw-run", type=Path, required=True)
    ap.add_argument("--prescreen-model", type=Path, required=True)
    ap.add_argument("--raw-model", type=Path, required=True)
    ap.add_argument("--property-dir", type=Path, required=True)
    ap.add_argument("--sand-mask", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    original = joblib.load(args.prescreen_model)
    raw = joblib.load(args.raw_model)
    runs = {}
    for label, path in (("prescreen", args.prescreen_run), ("raw", args.raw_run)):
        with np.load(path / "ckpt_gen0000.npz") as z:
            initial = np.asarray(z["pop"], dtype=int)
        with np.load(path / "ckpt_final_population.npz") as z:
            final = np.asarray(z["pop"], dtype=int)
            front = np.asarray(z["full_pop"], dtype=int)
        runs[label] = {"initial": initial, "final": final, "front": front}
    assert np.array_equal(runs["prescreen"]["initial"], runs["raw"]["initial"])

    prop = args.property_dir
    perm = read_property_txt(prop / "property_260421_all_perm.txt")
    poro = read_property_txt(prop / "property_260421_all_poro.txt")
    thick = read_cmg_txt(prop / "2026_0421_1day_property Grid Thickness 2025-Jan-01.txt")
    with np.load(args.sand_mask, allow_pickle=False) as z:
        sand = np.asarray(z["sand_mask_kij"], dtype=bool)
    report = {"same_initial_population": True, "models": {}}
    for label, pops in runs.items():
        report["models"][label] = {}
        for phase, pop in pops.items():
            geo = layout_metrics(pop, perm, poro, thick, sand)
            predictions = {"prescreen": original.predict(pop), "raw": raw.predict(pop)}
            entry = {"n": len(pop), "geology_median": summarize(geo, slice(None))}
            for key, pred in predictions.items():
                entry[f"{key}_predicted_objectives_median"] = np.median(pred, axis=0).tolist()
            own = predictions[label]
            for objective in range(3):
                keep = np.argsort(own[:, objective])[:max(1, len(pop) // 5)]
                entry[f"low_F{objective+1}_geology_median"] = summarize(geo, keep)
            report["models"][label][phase] = entry
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
