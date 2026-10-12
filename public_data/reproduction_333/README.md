# Derived inputs for the 333-feature surrogate

This directory contains the data needed to run the reported surrogate-training and NSGA-III commands in
`REPRODUCIBILITY_RELEASE.md` without access to the original geological model or CMG files.

| File | Contents |
|---|---|
| `training_prescreen_fields.npz` | Injection and production plan-view screening maps, valid-cell mask, total-thickness map, and grid dimensions |
| `prescreen_maps/static_sideview_fields_no_pressure_threshold.npz` | Three derived 3-D fields sampled to form the 218 side-view descriptors, plus interval bounds |
| `training_responses_684.npz` | Layout coordinates, three GEM-derived response values, success mask, and pressure standard deviation/mean for component-wise F3 training |
| `search_seed_geometry.npz` | Layout coordinates, feasible-cell mask, grid dimensions, and cell size for the surrogate NSGA-III search |
| `verified_training_report.json` | Feature order, split indices, XGBoost parameters, file checksums, and training/holdout/CV metrics from a fresh run with these files |

The total-thickness map is the absolute difference between the model-grid top and bottom
surfaces. Those original surfaces are not included. The released 3-D fields are transformed
feature inputs and do not contain the original facies, porosity, or permeability grids.
The compact search seed contains no 3-D geometry or SIP data. Its fields match the corresponding
fields of the original search seed and reproduce the saved generation-0 checkpoint with seed 42.
It can be rebuilt from the other released arrays by running
`python validation_tools/build_public_search_seed.py` from the repository root.
The released optimizer routes all stochastic choices through that seed and records the
generator and adaptive JADE states in new checkpoints for exact continuation.

The 684-row table contains 679 successful, finite training layouts. With the published
command, 543 rows enter the training split and 136 the holdout split; five-fold CV uses all
679 successful rows. The published inputs reproduce holdout R2 values of 0.9307, 0.9033,
and 0.8836 for F1, F2, and F3, respectively.
