# Reviewer 11: score-free geological test

## Question and design

The prescreen maps combine geological and operating information into injection/production potential. A rise in those map values after search is a description of selected layouts, not independent evidence of geological control. This analysis tests the geological story without using either prescreen score in the test itself.

Two complementary comparisons were performed on the same Shenhua field model:

1. **GEM responses against unscored geology.** The 679 completed layouts were joined by `sample_index` to the original permeability, thickness, and DMatlas sand-facies analyses. F1, F2, pressure-increment mean, pressure-increment standard deviation, and F3 are from GEM, not a surrogate. Well spacing and source-sink separation were calculated directly from the 20 well coordinates. Rank associations were recomputed after controlling the listed geometric/property covariates; 95% intervals use 1,000 layout-level bootstrap resamples. These are associations across the existing simulated layouts.
2. **No-score surrogate/search.** A 363-input model uses 20 coordinates, 47 common layout descriptors, separate well-column/local summaries of porosity, horizontal permeability, cell thickness, initial pressure, and grid-top depth. It does not use injection/production J maps, pressure-safety scores, or composite mobility fields. The five original fields are sampled and summarized, not passed as an uncompressed 3-D grid. The reported prescreen model has 333 inputs. Both use the same 679 successful layouts, 543/136 split, five folds, XGBoost settings, and 240-member/80-generation NSGA-III protocol with the same initial population and random seed. The brine viscosity is spatially constant; temperature and the modeled CO2 viscosity are functions of depth and initial pressure. No new GEM simulation was run.

## Full-physics geological relationships

| Relation | Partial rank r (95% bootstrap interval) | Physical reading |
|---|---:|---|
| Production-column integrated `kh` vs F1 | -0.875 (-0.896, -0.851) | Greater producer-side transmissive capacity accompanies smaller peak pressure rise. |
| Injection 500-m local sand fraction vs F2, controlling injection `kh` and geometry | -0.646 (-0.697, -0.590) | Lateral sand-body organization contributes beyond column capacity and injector spacing. |
| Injection longest low-`k` run vs F2, controlling injection `kh` and geometry | +0.324 (+0.249, +0.405) | Vertical interruption accompanies a larger projected plume. |
| Injection longest continuous facies-sand interval vs F2, controlling injection `kh` and geometry | -0.155 (-0.235, -0.080) | The direct facies measure retains the same direction, with a smaller independent association. |
| Injector spacing vs F2, controlling injection `kh` and vertical interruption | +0.215 (+0.134, +0.286) | Footprint overlap contributes alongside geology. |
| Production-column integrated `kh` vs pressure-increment mean | -0.838 (-0.867, -0.807) | The producer-side geological state organizes the regional pressure background. |
| Injection-production centroid offset vs pressure-increment SD | +0.783 (+0.749, +0.815) | Source-sink spatial mismatch organizes pressure contrasts. |
| Production-column integrated `kh` vs F3 | +0.657 (+0.609, +0.699) | Lower regional mean pressure can raise the relative spatial variability measured by F3. |

All figures and correlations above use original geological/coordinate fields and GEM outcomes only. F3 was checked as `pressure_std_mpa / pressure_mean_mpa` for every layout (maximum numerical difference below `7e-16`). Across source-sink centroid-offset quartiles, median pressure SD rises from 0.626 to 1.352 MPa while median mean pressure changes much less (1.467 to 1.526 MPa). The ratio therefore responds to both pressure relief and the organization of local pressure contrasts.

The 6-mD connected-component pair count has an unadjusted association with F1, but after controlling production `kh` and geometry its partial rank correlation is -0.038 (95% interval -0.112 to +0.034). The available component graph is almost saturated (428 of 679 layouts have all 10 producer pairs sharing a thresholded component). Thus this **particular graph metric does not independently quantify hydraulic network reach**. The full-physics case comparisons remain the evidence for the paper's stronger network-scale flow interpretation.

## Input comparison and score-free search

| Holdout R2 | Prescreen, 333 inputs | Separate original fields, 363 inputs |
|---|---:|---:|
| F1 | 0.931 | 0.901 |
| F2 | 0.903 | 0.800 |
| F3 | 0.884 | 0.874 |

Five-fold mean R2 is 0.914/0.899/0.838 for the prescreen model and 0.892/0.816/0.813 for the separate-field model. At 136 training layouts (three fixed subsamples), mean holdout R2 is 0.840/0.861/0.785 versus 0.782/0.743/0.765. At 272 layouts it is 0.902/0.894/0.854 versus 0.869/0.775/0.838. The largest and persistent predictive gain is for plume footprint, while F3 accuracy is similar. The comparison demonstrates the value of the geological representation under this sample budget, not that the original geological fields lack the underlying information.

The score-free search recovers the geological directions of the reported search. In each model's lowest-F1 20%, median production integrated `kh` increases from about 595 to 632 mD m with prescreen inputs, and from 594 to 645 mD m with separate-field inputs. In the lowest-F2 20%, injection longest continuous sand rises from 151 to 178 m and from 151 to 163 m, respectively; injection spacing decreases in both. In the lowest-F3 20%, source-sink centroid offset falls from 1,347 to 362 m and from 1,316 to 266 m, respectively. These search statistics are geological properties of surrogate-selected layouts; the independent GEM analysis above establishes the associated physical responses. Candidate layouts from the two searches were not newly simulated in GEM.

## Files and provenance

- `gem_scorefree_associations.csv`, `gem_scorefree_quartiles.csv`, and the `gem_*.png` figures: direct GEM/geology analysis.
- `primitive_input_comparison.json`, `primitive_learning_curve.png`: matched-input and sample-budget comparison.
- `primitive_search_geology.json`, `search_geological_directions.csv`, `primitive_nsga3/`: score-free search diagnosis.
- `primitive_full_surrogate.joblib`: fitted score-free surrogate for reproducibility; load with the accompanying extractor modules on Python's import path.
- `reviewer11_independent_analysis.py`, `reviewer11_input_comparison.py`, `reviewer11_primitive_features.py`, `reviewer11_search_summary.py`: analysis scripts. The input comparison also imports `raw_geology_features.py` and training helpers from the local `jcp_public_repository`.

Input sources: `jcp_public_repository/public_data/training_responses_684.npz`; `F:/CODE1/property_CMG` property text exports; and `F:/CODE1/_organized_active_work_20260817/06_verified_sandbody_analysis_0_683` verified sandbody/petrophysical analyses. The geological grid and property exports remain under their existing project confidentiality restrictions and are not duplicated here. No paper or supplementary manuscript file was changed.
