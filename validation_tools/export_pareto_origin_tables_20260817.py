from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"F:\CODE1\_organized_active_work_20260817"
    r"\05_nsga3_current_0_683_20260817"
)
ANALYSIS_DIR = ROOT / "final_analysis_cases_18"
OUT_DIR = ANALYSIS_DIR / "origin_pareto_tables"

STAGES = [
    ("Initial_Gen0", 0, ROOT / "ckpt_gen0000.npz"),
    ("Middle_Gen40", 40, ROOT / "ckpt_gen0040.npz"),
    ("Final_Gen80", 80, ROOT / "ckpt_gen0080.npz"),
]


def layout_key(layout: np.ndarray) -> tuple:
    layout = np.rint(np.asarray(layout, dtype=float)).astype(int)
    inj = tuple(sorted(tuple(map(int, row)) for row in layout[:10].reshape(5, 2)))
    prod = tuple(sorted(tuple(map(int, row)) for row in layout[10:].reshape(5, 2)))
    return inj, prod


def deduplicate(layouts: np.ndarray, objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    retained = {}
    for index, (layout, objective) in enumerate(zip(layouts, objectives)):
        key = layout_key(layout)
        previous = retained.get(key)
        if previous is None or tuple(objective) < tuple(objectives[previous]):
            retained[key] = index
    indices = np.asarray(sorted(retained.values()), dtype=int)
    return layouts[indices], objectives[indices]


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    mask = np.ones(len(objectives), dtype=bool)
    for index in range(len(objectives)):
        if not mask[index]:
            continue
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated[index] = False
        if np.any(dominated):
            mask[index] = False
    return mask


def objective_key(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(values, dtype=float), 12))


def load_selected() -> pd.DataFrame:
    selected = pd.read_csv(ANALYSIS_DIR / "analysis_cases.csv")
    objective_columns = [
        "F1_pressure_rise",
        "F2_plume_area_sg0p1_km2",
        "F3_pressure_CV_std_over_mean",
    ]
    if selected["full_pareto_index"].nunique() != 18:
        raise ValueError("Selected layouts contain duplicate Pareto indices")
    if selected[objective_columns].round(12).drop_duplicates().shape[0] != 18:
        raise ValueError("Selected points contain duplicate objective coordinates")
    selected["point_label"] = selected.apply(
        lambda row: f"S{int(row['case_id']):02d}_{row['role']}", axis=1
    )
    selected["F1_pressure_rise_pct"] = selected["F1_pressure_rise"] * 100.0
    return selected


def stage_front(
    stage: str,
    generation: int,
    checkpoint: Path,
    selected_lookup: dict[tuple[float, float, float], dict],
) -> pd.DataFrame:
    data = np.load(checkpoint, allow_pickle=False)
    layouts = np.rint(np.asarray(data["pop"], dtype=float)).astype(int)
    objectives = np.asarray(data["objs"], dtype=float)
    finite = np.isfinite(objectives).all(axis=1) & (objectives < 1.0e20).all(axis=1)
    layouts, objectives = deduplicate(layouts[finite], objectives[finite])
    mask = nondominated_mask(objectives)
    layouts = layouts[mask]
    objectives = objectives[mask]

    # Remove objective-coordinate duplicates that would overlap in Origin.
    unique_indices = []
    seen = set()
    for index, objective in enumerate(objectives):
        key = objective_key(objective)
        if key not in seen:
            unique_indices.append(index)
            seen.add(key)
    layouts = layouts[unique_indices]
    objectives = objectives[unique_indices]
    order = np.lexsort((objectives[:, 2], objectives[:, 1], objectives[:, 0]))
    layouts = layouts[order]
    objectives = objectives[order]

    rows = []
    for point_number, (layout, objective) in enumerate(
        zip(layouts, objectives), start=1
    ):
        selected = selected_lookup.get(objective_key(objective), {})
        rows.append(
            {
                "Stage": stage,
                "Generation": generation,
                "Front_point_id": point_number,
                "F1_pressure_rise": float(objective[0]),
                "F1_pressure_rise_pct": float(objective[0] * 100.0),
                "F2_plume_area_km2": float(objective[1]),
                "F3_pressure_heterogeneity": float(objective[2]),
                "Selected": int(bool(selected)),
                "Selected_case_id": selected.get("case_id", ""),
                "Selected_label": selected.get("point_label", ""),
                "Selected_group": selected.get("group", ""),
                "Selected_role": selected.get("role", ""),
                "Machine": selected.get("machine", ""),
                "Layout_key": json.dumps(layout_key(layout)),
            }
        )
    return pd.DataFrame(rows)


def saved_checkpoints() -> list[tuple[int, Path]]:
    checkpoints = []
    for path in ROOT.glob("ckpt_gen*.npz"):
        match = re.fullmatch(r"ckpt_gen(\d{4})\.npz", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    return checkpoints


def full_population_table(
    generation: int,
    checkpoint: Path,
    selected_lookup: dict[tuple[float, float, float], dict],
) -> pd.DataFrame:
    data = np.load(checkpoint, allow_pickle=False)
    layouts = np.rint(np.asarray(data["pop"], dtype=float)).astype(int)
    objectives = np.asarray(data["objs"], dtype=float)
    feasible = np.isfinite(objectives).all(axis=1) & (objectives < 1.0e20).all(axis=1)
    pareto = np.zeros(len(objectives), dtype=bool)
    pareto[feasible] = nondominated_mask(objectives[feasible])

    layout_keys = [layout_key(layout) for layout in layouts]
    layout_counts = {}
    for key in layout_keys:
        layout_counts[key] = layout_counts.get(key, 0) + 1
    objective_keys = [objective_key(objective) for objective in objectives]
    objective_counts = {}
    for key in objective_keys:
        objective_counts[key] = objective_counts.get(key, 0) + 1

    rows = []
    for point_id, (layout, objective, key) in enumerate(
        zip(layouts, objectives, layout_keys), start=1
    ):
        selected = (
            selected_lookup.get(objective_key(objective), {})
            if generation == 80
            else {}
        )
        rows.append(
            {
                "Generation": generation,
                "Population_point_id": point_id,
                "F1_pressure_rise": float(objective[0]),
                "F1_pressure_rise_pct": float(objective[0] * 100.0),
                "F2_plume_area_km2": float(objective[1]),
                "F3_pressure_heterogeneity": float(objective[2]),
                "Is_feasible": int(feasible[point_id - 1]),
                "Is_Pareto": int(pareto[point_id - 1]),
                "Layout_duplicate_count": int(layout_counts[key]),
                "Objective_duplicate_count": int(
                    objective_counts[objective_key(objective)]
                ),
                "Selected": int(bool(selected)),
                "Selected_case_id": selected.get("case_id", ""),
                "Selected_label": selected.get("point_label", ""),
                "Selected_group": selected.get("group", ""),
                "Selected_role": selected.get("role", ""),
                "Machine": selected.get("machine", ""),
                "Layout_key": json.dumps(key),
            }
        )
    return pd.DataFrame(rows)


def build_wide(fronts: list[pd.DataFrame]) -> pd.DataFrame:
    columns = {}
    for frame in fronts:
        stage = frame["Stage"].iloc[0]
        short = stage.replace("Initial_", "G0_").replace("Middle_", "G40_").replace(
            "Final_", "G80_"
        )
        columns[f"{short}F1"] = pd.Series(frame["F1_pressure_rise"].to_numpy())
        columns[f"{short}F2_km2"] = pd.Series(frame["F2_plume_area_km2"].to_numpy())
        columns[f"{short}F3"] = pd.Series(frame["F3_pressure_heterogeneity"].to_numpy())
    return pd.DataFrame(columns)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = load_selected()
    selected_lookup = {
        objective_key(
            row[
                [
                    "F1_pressure_rise",
                    "F2_plume_area_sg0p1_km2",
                    "F3_pressure_CV_std_over_mean",
                ]
            ].to_numpy(dtype=float)
        ): row.to_dict()
        for _, row in selected.iterrows()
    }

    fronts = [
        stage_front(stage, generation, checkpoint, selected_lookup)
        for stage, generation, checkpoint in STAGES
    ]
    long_table = pd.concat(fronts, ignore_index=True)
    wide_table = build_wide(fronts)

    checkpoint_list = saved_checkpoints()
    population_frames = [
        full_population_table(generation, checkpoint, selected_lookup)
        for generation, checkpoint in checkpoint_list
    ]
    all_populations = pd.concat(population_frames, ignore_index=True)
    all_saved_fronts = pd.concat(
        [
            stage_front(
                f"Gen{generation}", generation, checkpoint, selected_lookup
            )
            for generation, checkpoint in checkpoint_list
        ],
        ignore_index=True,
    )
    generation_summary = pd.DataFrame(
        [
            {
                "Generation": generation,
                "Population_points": int(len(frame)),
                "Feasible_points": int(frame["Is_feasible"].sum()),
                "Pareto_points_with_duplicates": int(frame["Is_Pareto"].sum()),
                "Unique_layouts": int(frame["Layout_key"].nunique()),
                "Unique_objective_coordinates": int(
                    frame[
                        [
                            "F1_pressure_rise",
                            "F2_plume_area_km2",
                            "F3_pressure_heterogeneity",
                        ]
                    ].round(12).drop_duplicates().shape[0]
                ),
                "Unique_exported_front_points": int(
                    (all_saved_fronts["Generation"] == generation).sum()
                ),
            }
            for (generation, _), frame in zip(checkpoint_list, population_frames)
        ]
    )

    selected_columns = [
        "case_id",
        "point_label",
        "full_pareto_index",
        "group",
        "role",
        "machine",
        "F1_pressure_rise",
        "F1_pressure_rise_pct",
        "F2_plume_area_sg0p1_km2",
        "F3_pressure_CV_std_over_mean",
        "relative_dat_path",
    ]
    selected_table = selected[selected_columns].copy()
    selected_table = selected_table.rename(
        columns={
            "F2_plume_area_sg0p1_km2": "F2_plume_area_km2",
            "F3_pressure_CV_std_over_mean": "F3_pressure_heterogeneity",
        }
    )

    long_table.to_csv(OUT_DIR / "pareto_fronts_origin_long.csv", index=False)
    wide_table.to_csv(OUT_DIR / "pareto_fronts_origin_wide.csv", index=False)
    selected_table.to_csv(OUT_DIR / "selected_18_points_origin.csv", index=False)
    all_populations.to_csv(
        OUT_DIR / "all_saved_generations_population_origin.csv", index=False
    )
    all_saved_fronts.to_csv(
        OUT_DIR / "all_saved_generations_pareto_origin.csv", index=False
    )
    generation_summary.to_csv(
        OUT_DIR / "all_saved_generations_summary.csv", index=False
    )
    for frame in fronts:
        frame.to_csv(
            OUT_DIR / f"{frame['Stage'].iloc[0]}_front.csv", index=False
        )

    instructions = pd.DataFrame(
        [
            {
                "Item": "Objective direction",
                "Description": "F1, F2, and F3 are minimized.",
            },
            {
                "Item": "Initial stage",
                "Description": "Gen 0 is the evaluated initial population before the first evolutionary update.",
            },
            {
                "Item": "Origin long table",
                "Description": "Use F1, F2, F3 as XYZ and Stage as the grouping column.",
            },
            {
                "Item": "Selected markers",
                "Description": "Use selected_18_points_origin.csv and label points with point_label.",
            },
            {
                "Item": "All saved populations",
                "Description": "Population_all_saved contains all 240 points at Gen 0, 5, 10, ..., 80; filter with Is_Pareto when needed.",
            },
            {
                "Item": "Checkpoint interval",
                "Description": "The optimizer saved every five generations; unsaved generations cannot be reconstructed.",
            },
        ]
    )

    workbook = OUT_DIR / "pareto_fronts_and_selected_points_origin.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        long_table.to_excel(writer, sheet_name="Fronts_long", index=False)
        wide_table.to_excel(writer, sheet_name="Fronts_wide", index=False)
        fronts[0].to_excel(writer, sheet_name="Initial_Gen0", index=False)
        fronts[1].to_excel(writer, sheet_name="Middle_Gen40", index=False)
        fronts[2].to_excel(writer, sheet_name="Final_Gen80", index=False)
        selected_table.to_excel(writer, sheet_name="Selected_18", index=False)
        all_populations.to_excel(writer, sheet_name="Population_all_saved", index=False)
        all_saved_fronts.to_excel(writer, sheet_name="Pareto_all_saved", index=False)
        generation_summary.to_excel(writer, sheet_name="Generation_summary", index=False)
        instructions.to_excel(writer, sheet_name="README", index=False)

    summary = {
        frame["Stage"].iloc[0]: int(len(frame)) for frame in fronts
    }
    summary["selected_unique_points"] = int(len(selected_table))
    summary["saved_generations"] = [generation for generation, _ in checkpoint_list]
    summary["all_saved_population_points"] = int(len(all_populations))
    summary["all_saved_pareto_rows"] = int(len(all_saved_fronts))
    (OUT_DIR / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved Origin tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
