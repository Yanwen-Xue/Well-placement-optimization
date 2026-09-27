# Geological Prescreening Surrogate Optimization for CO2 Storage

This repository contains the Python workflow used for geological-prescreening-assisted
well-placement optimization in a heterogeneous CO2 storage reservoir model.

The workflow includes:

- construction of geological prescreening maps for CO2 injection and brine production;
- Latin hypercube sampling of feasible well layouts;
- extraction of objective values from full-physics CMG simulation outputs;
- XGBoost surrogate-model training;
- surrogate-assisted NSGA-III multi-objective optimization;
- validation and ablation utilities for prescreen-derived surrogate inputs.

## Objective Functions

The optimization problem uses three minimization objectives:

- `F1`: peak relative increase in average reservoir pressure;
- `F2`: horizontal projected CO2 plume-footprint area, calculated from grid columns where the maximum gas saturation in the CO2 injection interval is at least 0.10;
- `F3`: normalized pressure-buildup heterogeneity, calculated as the pore-volume-weighted standard deviation of pressure buildup divided by the pore-volume-weighted mean pressure buildup in the CO2 injection interval.

Injection wells are constrained by a location-specific bottom-hole-pressure limit based on
90% of the Eaton fracture-pressure estimate.

## Repository Structure

```text
.
|-- cmg_runner.py
|-- injectivity_prescreening_map.py
|-- lhs_sampler.py
|-- objectives.py
|-- optimizer.py
|-- surrogate.py
|-- feature_333.py
|-- component_f3_surrogate.py
`-- validation_tools/
    |-- build_static_sideview_fields.py
    |-- build_optimizer_seeds.py
    |-- train_component_f3_surrogate.py
    |-- append_lhs_samples_and_export.py
    |-- build_lhs_results_from_export.py
    |-- export_lhs_group.py
    |-- export_pareto_validation_groups.py
    |-- run_lhs_doe.py
    |-- train_surrogate.py
    `-- validate_prescreen_utility.py
```

## Required External Files

The geological model, CMG input files, and SR3 simulation outputs are not included because
they contain confidential reservoir information and commercial simulator outputs.

The reported 333-feature model requires the plan-view prescreen arrays in `opt_seeds.npz`
and the normalized, layer-resolved static side-view arrays in
`prescreen_maps/static_sideview_fields_no_pressure_threshold.npz`. These spatial arrays
are derived from the restricted geological model and are not distributed here. The public
`public_data/training_responses_684.npz` contains only layout coordinates, objective values,
success flags, and pressure standard-deviation/mean targets (679 successful layouts).

To regenerate the prescreen inputs from authorized CMG property exports, provide the
`Pressure`, `Grid Thickness`, `Grid Top`, `Grid Centroid X/Y`, `Gas Viscosity`,
`Water Viscosity`, and `Water Saturation` exports and the porosity/permeability grids.
The filenames below match the property exports used for the paper. CMG itself is needed
only for new full-physics simulations.

```powershell
$private = 'C:\path\to\restricted_inputs'
$env:CMG_PROPERTY_DIR = Join-Path $private 'property_CMG'
$env:CMG_PROPERTY_PREFIX = '2026_0421_1day_property '
$env:CMG_PROPERTY_SUFFIX = ' 2025-Jan-01.txt'
$env:CMG_POROSITY_FILE = Join-Path $env:CMG_PROPERTY_DIR 'property_260421_all_poro.txt'
$env:CMG_PERMEABILITY_FILE = Join-Path $env:CMG_PROPERTY_DIR 'property_260421_all_perm.txt'
$env:PRESCREEN_OUTPUT_DIR = $private
python injectivity_prescreening_map.py
python validation_tools/build_static_sideview_fields.py --property-dir $env:CMG_PROPERTY_DIR --seeds (Join-Path $private 'opt_seeds.npz') --out (Join-Path $private 'prescreen_maps/static_sideview_fields_no_pressure_threshold.npz')
```

For new full-physics cases, additionally configure the commercial simulator:

```powershell
$env:CMG_EXE="path\to\gm202420.exe"
$env:CMG_TEMPLATE_DAT="path\to\template.dat"
$env:CMG_SIP_FILE="path\to\model.sip"
```

## Reported Surrogate And Search

The model uses 20 well coordinates and 313 constructed descriptors: 333 inputs in total.
`feature_333.py` constructs the static well-column and layout descriptors;
`surrogate.py` constructs the plan-view descriptors; and
`validation_tools/train_surrogate.py` assembles them in
`prescreen_geometry_local_static_well_triview` mode. The reported F1 and F2 heads
predict directly. The F3 head fits log pressure standard deviation and log pressure
mean separately, then takes their ratio. The general-purpose
`validation_tools/train_surrogate.py` command is **not** the reported model.

Install dependencies with `python -m pip install -r requirements.txt`. The reported
training command, run from the repository root after the prescreen inputs above exist, is:

```powershell
python validation_tools/train_component_f3_surrogate.py `
  --results public_data/training_responses_684.npz `
  --seeds (Join-Path $private 'opt_seeds.npz') `
  --model-out (Join-Path $private 'surrogate_333.joblib') `
  --report-out (Join-Path $private 'surrogate_333_metrics.json') `
  --feature-mode prescreen_geometry_local_static_well_triview `
  --test-size 0.20 --split-seed 42 --cv-splits 5 `
  --n-jobs 16 --n-estimators 800 --max-depth 2 --learning-rate 0.03 `
  --subsample 0.90 --colsample-bytree 0.90 --reg-lambda 5.0 --min-child-weight 3.0
```

Training uses CPU histogram trees, XGBoost `random_state=42`, zero L1 penalty,
no early stopping, and all 679 successful layouts
for the final fitted model. The 543/136 split gives holdout R2 values of 0.931, 0.903,
and 0.884 for F1/F2/F3; five-fold CV means are 0.914, 0.899, and 0.838. The
report JSON records the full ordered feature list and the train/test sample indices.
The verified environment was Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3,
scikit-learn 1.7.2, XGBoost 3.2.0, and joblib 1.5.2.

Assemble the original 684-layout seed pool from the published coordinates and local
prescreen maps, then run the paper-sized NSGA-III search:

```powershell
python validation_tools/build_optimizer_seeds.py `
  --responses public_data/training_responses_684.npz `
  --maps (Join-Path $private 'opt_seeds.npz') `
  --out (Join-Path $private 'optimizer_seeds_684.npz')
$env:OPT_RUNS_DIR = Join-Path $private 'nsga3_runs'
python optimizer.py `
  --seeds (Join-Path $private 'optimizer_seeds_684.npz') `
  --surrogate (Join-Path $private 'surrogate_333.joblib') `
  --pop 240 --gen 80 --ref-p 12 --final-k 16 --parallel 32 `
  --use-de --use-opt-perm-align --no-early-stop --seed 42
```

The search uses 91 Das--Dennis reference directions, adaptive differential evolution,
and no early stopping. The code sets initial DE scale/crossover to 0.50/0.90,
adaptation rate to 0.10, best-parent fraction to 0.10, and checkpoints every five
generations. Boundary retreat is 1000 m; injection--injection and
production--production spacing are at least 200 m, and injection--production spacing
is at least 300 m. `--seed 42` makes this rerun deterministic; the historical
optimizer run did not record its RNG seed, so individual Pareto members need not be
identical to the published run. The reported final 240 layouts and responses are in
`public_data/Supplementary_Data_Pareto_240_Layouts_Three_Responses.csv`.

## Data Availability

The geological model, reservoir-model input files, petrophysical property grids, and
full-physics simulation outputs used in this study are derived from the Shenhua Ordos CO2
geological storage demonstration project and are subject to confidentiality restrictions.
Therefore, these data are not publicly available.

## Public paper-case tables

Paper-facing derived tables for the eight selected layouts are provided in
`public_data/paper_cases_1_8/`. They include the case mapping, well coordinates,
0--80 year response summaries, paired differences, Pareto-screening records, and
matched-layout controls. The original CMG `.sr3` result files are documented in
`sr3_manifest.csv` but are not included because each file is approximately 9--10 GB.
SIP files, geological model files, property grids, and CMG input files remain excluded.

The complete derived response tables are provided in `public_data/`. The file
`Supplementary_Data_679_Layouts_Three_Responses.csv` contains the five injector and five
producer well locations as zero-based model-grid I/J indices, the three response metrics for
all 679 completed layouts, and the pressure-peak time. The file
`Supplementary_Data_Pareto_240_Layouts_Three_Responses.csv` contains the same well-location
fields and the three surrogate-predicted responses for the 240 non-dominated layouts. These
tables contain no geological model, CMG input, SIP file, or SR3 output.

The non-spatial interval statistics reported in manuscript Table 1 and a concise
description of flow-model settings are in `public_data/model_interval_summary.csv`
and `public_data/model_parameter_notes.md`. These files do not include spatial
property grids or site measurements.
