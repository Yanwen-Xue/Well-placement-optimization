from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\CODE1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cmg_runner  # noqa: E402
from validation_tools.train_surrogate import build_feature_extractor  # noqa: E402


NSGA_DIR = (
    ROOT
    / "_organized_active_work_20260817"
    / "05_nsga3_current_0_683_20260817"
)
CHECKPOINT = NSGA_DIR / "ckpt_final_population.npz"
SEEDS = ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz"
OUT_DIR = NSGA_DIR / "final_analysis_cases_18"
SIP = ROOT / "cmg" / "2026_0421.sip"
RANDOM_SEED = 42


def normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0)
    span = np.maximum(np.ptp(values, axis=0), 1.0e-12)
    return (values - minimum) / span


def select_anchors(objectives: np.ndarray) -> list[tuple[int, str, str]]:
    normalized = normalize(objectives)
    anchors = [
        (int(np.argmin(objectives[:, 0])), "anchor", "F1_extreme"),
        (int(np.argmin(objectives[:, 1])), "anchor", "F2_extreme"),
        (int(np.argmin(objectives[:, 2])), "anchor", "F3_extreme"),
        (
            int(np.argmin(np.linalg.norm(normalized - 0.5, axis=1))),
            "anchor",
            "balanced_center",
        ),
    ]
    if len({index for index, _, _ in anchors}) != 4:
        raise ValueError("Response anchors are not unique")
    return anchors


def best_pair(
    objectives: np.ndarray,
    target: int,
    used: set[int],
) -> tuple[int, int, np.ndarray]:
    span = np.maximum(np.ptp(objectives, axis=0), 1.0e-12)
    candidates = []
    for left in range(len(objectives)):
        if left in used:
            continue
        for right in range(left + 1, len(objectives)):
            if right in used:
                continue
            delta = np.abs(objectives[left] - objectives[right])
            if target == 2:
                valid = delta[0] <= 0.006 and delta[1] <= 0.18 and delta[2] >= 0.18
                controls = (0, 1)
            elif target == 1:
                valid = delta[0] <= 0.008 and delta[1] >= 0.80 and delta[2] <= 0.06
                controls = (0, 2)
            else:
                valid = delta[0] >= 0.035 and delta[1] <= 0.18 and delta[2] <= 0.06
                controls = (1, 2)
            if not valid:
                continue
            control_distance = np.linalg.norm(delta[list(controls)] / span[list(controls)])
            score = (delta[target] / span[target]) / (control_distance + 0.02)
            candidates.append((score, left, right, delta))
    if not candidates:
        raise RuntimeError(f"No valid matched pair found for F{target + 1}")
    _, left, right, delta = max(candidates, key=lambda row: row[0])
    return left, right, delta


def select_matched_pairs(
    objectives: np.ndarray,
    used: set[int],
) -> tuple[list[tuple[int, str, str]], list[dict[str, float | str]]]:
    cases = []
    differences = []
    for target, name in ((2, "F3_pair"), (1, "F2_pair"), (0, "F1_pair")):
        left, right, delta = best_pair(objectives, target, used)
        ordered = sorted((left, right), key=lambda index: objectives[index, target])
        cases.extend(
            [
                (ordered[0], name, f"F{target + 1}_low"),
                (ordered[1], name, f"F{target + 1}_high"),
            ]
        )
        used.update((left, right))
        differences.append(
            {
                "pair": name,
                "delta_F1": float(delta[0]),
                "delta_F2_km2": float(delta[1]),
                "delta_F3": float(delta[2]),
            }
        )
    return cases, differences


def select_response_coverage(
    objectives: np.ndarray,
    used: set[int],
    count: int = 4,
) -> list[tuple[int, str, str]]:
    normalized = normalize(objectives)
    selected = []
    seed_indices = sorted(used)
    while len(selected) < count:
        reference = seed_indices + selected
        distances = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[reference][None, :, :], axis=2
            ),
            axis=1,
        )
        distances[list(used) + selected] = -1.0
        selected.append(int(np.argmax(distances)))
    used.update(selected)
    return [
        (index, "response_coverage", f"coverage_{rank}")
        for rank, index in enumerate(selected, start=1)
    ]


def geological_feature_matrix(layouts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    extractor, report = build_feature_extractor(
        SEEDS, "prescreen_geometry_local_static_well_triview"
    )
    values = np.asarray(extractor(layouts), dtype=float)
    names = list(report["feature_names"])
    if values.shape[1] != len(names):
        raise ValueError("Geological feature names and values are misaligned")

    prefixes = (
        "inj_J_",
        "prod_J_",
        "inj_thickness_",
        "injw_static_",
        "prodw_static_",
        "inj_prod_static_",
        "static_inj_",
    )
    keep = [index for index, name in enumerate(names) if name.startswith(prefixes)]
    selected_values = values[:, keep]
    selected_names = [names[index] for index in keep]
    finite = np.isfinite(selected_values).all(axis=0)
    variable = np.nanstd(selected_values, axis=0) > 1.0e-12
    mask = finite & variable
    return selected_values[:, mask], [
        name for name, retain in zip(selected_names, mask) if retain
    ]


def select_geological_archetypes(
    layouts: np.ndarray,
    used: set[int],
    count: int = 4,
) -> tuple[list[tuple[int, str, str]], dict]:
    geological, names = geological_feature_matrix(layouts)
    standardized = StandardScaler().fit_transform(geological)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_SEED)
    scores = pca.fit_transform(standardized)
    clustering = KMeans(n_clusters=count, n_init=50, random_state=RANDOM_SEED)
    labels = clustering.fit_predict(scores)

    cases = []
    cluster_rows = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        order = members[
            np.argsort(
                np.linalg.norm(scores[members] - clustering.cluster_centers_[cluster], axis=1)
            )
        ]
        available = [int(index) for index in order if int(index) not in used]
        if not available:
            raise RuntimeError(f"No unused geological medoid in cluster {cluster}")
        chosen = available[0]
        used.add(chosen)
        cases.append(
            (chosen, "geological_archetype", f"cluster_{cluster + 1}_medoid")
        )

        cluster_mean = standardized[members].mean(axis=0)
        high = np.argsort(cluster_mean)[-5:][::-1]
        low = np.argsort(cluster_mean)[:5]
        cluster_rows.append(
            {
                "cluster": cluster + 1,
                "n_front_members": int(len(members)),
                "selected_full_pareto_index": chosen,
                "high_features": "; ".join(names[index] for index in high),
                "low_features": "; ".join(names[index] for index in low),
            }
        )

    metadata = {
        "n_input_geological_features": int(geological.shape[1]),
        "pca_components": int(scores.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "clustering": "KMeans on standardized PCA scores",
        "random_seed": RANDOM_SEED,
        "clusters": cluster_rows,
    }
    return cases, metadata


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    data = np.load(CHECKPOINT, allow_pickle=False)
    layouts = np.rint(np.asarray(data["full_pop"], dtype=float)).astype(int)
    objectives = np.asarray(data["full_objs"], dtype=float)

    anchors = select_anchors(objectives)
    used = {index for index, _, _ in anchors}
    matched, pair_differences = select_matched_pairs(objectives, used)
    coverage = select_response_coverage(objectives, used)
    archetypes, geological_metadata = select_geological_archetypes(layouts, used)
    specifications = anchors + matched + coverage + archetypes
    if len(specifications) != 18 or len({row[0] for row in specifications}) != 18:
        raise ValueError("Final analysis selection must contain 18 unique cases")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cmg").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SIP, OUT_DIR / "cmg" / SIP.name)
    cmg_runner.OPT_RUNS_DIR = str(OUT_DIR)

    cases = []
    for case_id, (full_index, group, role) in enumerate(specifications, start=1):
        layout = layouts[full_index]
        objective = objectives[full_index]
        inj = [tuple(map(int, row)) for row in layout[:10].reshape(5, 2)]
        prod = [tuple(map(int, row)) for row in layout[10:].reshape(5, 2)]
        case_name = f"case_{case_id:04d}_{group}_{role}"
        run_dir, dat_stem, inj_rates, capacities, shortfall = cmg_runner.make_run_dat(
            run_id=case_id,
            inj_wells=inj,
            prod_wells=prod,
            sim_years=80,
            gen_subdir=case_name,
            local_rid=1,
        )
        dat_path = Path(run_dir) / f"{dat_stem}.dat"
        cases.append(
            {
                "case_id": case_id,
                "full_pareto_index": full_index,
                "group": group,
                "role": role,
                "F1_pressure_rise": float(objective[0]),
                "F2_plume_area_sg0p1_km2": float(objective[1]),
                "F3_pressure_CV_std_over_mean": float(objective[2]),
                "inj_wells_0based": [list(well) for well in inj],
                "prod_wells_0based": [list(well) for well in prod],
                "inj_wells_1based": [[i + 1, j + 1] for i, j in inj],
                "prod_wells_1based": [[i + 1, j + 1] for i, j in prod],
                "inj_rates_surface_nm3_day": [float(v) for v in inj_rates],
                "inj_capacity_surface_nm3_day": [float(v) for v in capacities],
                "inj_shortfall_surface_nm3_day": float(shortfall),
                "relative_dat_path": dat_path.relative_to(OUT_DIR).as_posix(),
            }
        )

    manifest = {
        "source_checkpoint": str(CHECKPOINT),
        "source_surrogate": str(data["surrogate_path"].item()),
        "n_cases": len(cases),
        "selection_design": {
            "response_anchors": 4,
            "single_response_matched_pairs": 6,
            "response_space_coverage": 4,
            "prescreen_static_geological_archetypes": 4,
        },
        "pair_differences": pair_differences,
        "geological_sampling": geological_metadata,
        "cases": cases,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fields = [
        "case_id",
        "full_pareto_index",
        "group",
        "role",
        "F1_pressure_rise",
        "F2_plume_area_sg0p1_km2",
        "F3_pressure_CV_std_over_mean",
        "relative_dat_path",
    ]
    write_csv(OUT_DIR / "analysis_cases.csv", cases, fields)
    write_csv(
        OUT_DIR / "pair_differences.csv",
        pair_differences,
        ["pair", "delta_F1", "delta_F2_km2", "delta_F3"],
    )
    write_csv(
        OUT_DIR / "geological_archetypes.csv",
        geological_metadata["clusters"],
        [
            "cluster",
            "n_front_members",
            "selected_full_pareto_index",
            "high_features",
            "low_features",
        ],
    )
    print(f"Generated {len(cases)} final analysis DAT files in {OUT_DIR}")


if __name__ == "__main__":
    main()
