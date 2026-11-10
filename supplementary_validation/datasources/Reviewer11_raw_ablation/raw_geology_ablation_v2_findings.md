# Score-free geological-variable ablation (Reviewer 11)

## Input and protocol

The raw-only surrogate samples the Shenhua model's porosity, horizontal permeability, cell thickness, and **initial** pressure grids. It uses layer-wise column statistics from the injection (K41-80), production (K1-15), and intervening (K16-40) intervals; per-well target-interval means and standard deviations; and local means along both horizontal axes. It also receives the 20 well coordinates and the same 47 layout-geometry descriptors as the reported model. No injection/production J map, prescreen score, mobility composite, or full-physics response field enters the raw input. The input is a 323-variable tabular summary of the geological grids, not the uncompressed 3-D arrays. Facies labels are used only for independent post-search diagnosis.

The reported model uses 333 variables (20 coordinates, 48 plan-view descriptors, 218 side-view descriptors, 47 layout descriptors). Both models use the same 679 completed layouts, 543/136 split, five CV folds, four XGBoost heads, fixed parameters and seed 42. NSGA-III uses the same 240 initial layouts, 80 generations, 91 reference directions, and well constraints.

| R2 | Reported prescreen model: holdout / CV | Raw-only model: holdout / CV |
|---|---:|---:|
| F1 pressure rise | 0.931 / 0.914 | 0.898 / 0.890 |
| F2 plume footprint | 0.903 / 0.899 | 0.799 / 0.814 |
| F3 pressure heterogeneity | 0.884 / 0.838 | 0.874 / 0.809 |

## Geological diagnosis of the searches

The following medians describe the lowest 20% of each population for the stated model-predicted objective, evaluated on the original property and independently reconstructed sand-facies grids.

| Measure | Prescreen Gen0 -> Gen80 | Raw-only Gen0 -> Gen80 |
|---|---:|---:|
| Low-F1 production integrated kh (mD m) | 596 -> 632 | 596 -> 634 |
| Low-F1 production NTG | 0.893 -> 0.933 | 0.893 -> 0.947 |
| Low-F2 injection longest sand interval (m) | 151 -> 178 | 151 -> 174 |
| Low-F2 injection integrated kh (mD m) | 1304 -> 1391 | 1299 -> 1378 |
| Low-F2 mean injector spacing (m) | 4656 -> 2906 | 4656 -> 3645 |

Both representations find the same directions for production-side transmissive capacity and injection-side vertical sand continuity. These geological associations therefore do not require the J scores to appear in the search. The prescreen model predicts F2 substantially more accurately with a similarly sized input. This experiment does not establish which final layouts have better full-physics responses; no new GEM simulations were run. The original paper's interpretation of network-scale hydraulic connectivity and pressure-field organization still rests on its full-physics cases, not these raw summary statistics.

Artifacts: `raw_geology_ablation_v2.json`, `raw_geology_search_comparison_v2.json`, `ablation_raw_v2_nsga3/`, and the extractor/training/comparison scripts in `jcp_public_repository/`.
