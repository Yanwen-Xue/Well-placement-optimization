# Geological Prescreening Surrogate Optimization for CO2 Storage

This repository contains the Python workflow used for geological-prescreening-assisted
well-placement optimization in a heterogeneous CO2 storage reservoir model.

The workflow includes:

- construction of geological prescreening maps for CO2 injection and brine production;
- Latin hypercube sampling of feasible well layouts;
- extraction of objective values from full-physics CMG simulation outputs;
- XGBoost surrogate-model training;
- surrogate-assisted NSGA-III multi-objective optimization;
- validation and ablation utilities for the prescreen-derived surrogate inputs.

## Objective Functions

The final optimization problem uses three minimization objectives:

- `F1`: maximum relative increase in average pressure;
- `F2`: negative mobility-weighted gas-phase CO2 index;
- `F3`: PV-weighted spatial standard deviation of pressure buildup in the CO2 injection interval at the end of injection.

Injection wells are constrained by a location-specific bottom-hole-pressure limit based on
the fracture-pressure safety criterion.

## Repository Structure

```text
.
├── cmg_runner.py
├── injectivity_prescreening_map.py
├── lhs_sampler.py
├── objectives.py
├── optimizer.py
├── surrogate.py
└── validation_tools/
    ├── append_lhs_samples_and_export.py
    ├── build_lhs_results_from_export.py
    ├── export_lhs_group.py
    ├── export_pareto_validation_groups.py
    ├── run_lhs_doe.py
    ├── train_surrogate.py
    └── validate_prescreen_utility.py
```

## Required External Files

The geological model, CMG input files, and SR3 simulation outputs are not included because
they contain confidential reservoir information and commercial simulator outputs.

To run the full workflow, provide local files through environment variables:

```powershell
$env:CMG_EXE="path\to\gm202420.exe"
$env:CMG_MODEL_NAME="model"
$env:CMG_TEMPLATE_DAT="path\to\template.dat"
$env:CMG_SIP_FILE="path\to\model.sip"
$env:CMG_PROPERTY_DIR="path\to\property_files"
$env:CMG_PROPERTY_PREFIX="model_property "
$env:CMG_PROPERTY_SUFFIX=" 2025-Jan-01.txt"
$env:CMG_POROSITY_FILE="path\to\porosity.txt"
$env:CMG_PERMEABILITY_FILE="path\to\permeability.txt"
```

## Typical Workflow

1. Build prescreening maps:

```powershell
python injectivity_prescreening_map.py
```

2. Generate LHS well-layout seeds:

```powershell
python lhs_sampler.py --out opt_seeds_lhs.npz
```

3. Export or run full-physics simulation cases:

```powershell
python validation_tools/export_lhs_group.py --group-name lhs_cases --start 0 --stop 100
```

4. Build the training dataset from returned CMG outputs:

```powershell
python validation_tools/build_lhs_results_from_export.py --out opt_runs_lhs_doe/results_0_543.npz
```

5. Train the XGBoost surrogate:

```powershell
python validation_tools/train_surrogate.py --results opt_runs_lhs_doe/results_0_543.npz --model opt_runs_lhs_doe/surrogate_model_0_543.joblib --report opt_runs_lhs_doe/surrogate_metrics_0_543.json
```

6. Run NSGA-III with the trained surrogate:

```powershell
python optimizer.py --surrogate opt_runs_lhs_doe/surrogate_model_0_543.joblib
```

## Data Availability

The geological model, reservoir-model input files, petrophysical property grids, and
full-physics simulation outputs used in this study are derived from the Shenhua Ordos CO2
geological storage demonstration project and are subject to confidentiality restrictions.
Therefore, these data are not publicly available.
