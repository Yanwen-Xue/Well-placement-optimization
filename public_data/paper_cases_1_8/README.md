# Paper cases 1-8

This directory contains the public, paper-facing tables for the eight selected
full-physics layouts used in the manuscript.

The case numbering follows the paper:

| Paper case | Original analysis case | Role |
|---:|---:|---|
| 1 | 9 | F1 low-pressure member |
| 2 | 10 | F1 high-pressure member |
| 3 | 7 | clustered F2 low-area member |
| 4 | 8 | clustered F2 high-area member |
| 5 | 255 | distributed F2 high-area member |
| 6 | 312 | distributed F2 low-area member |
| 7 | 5 | F3 low-heterogeneity member |
| 8 | 6 | F3 high-heterogeneity member |

Files:

- `analysis_cases.csv`: pressure-peak, pressure-field, plume-area, and brine-production metrics at 30 and 80 years.
- `case_timeseries_0_80y.csv`: reservoir-average pressure and cumulative brine production at 0, 10, ..., 80 years.
- `well_coordinates.csv`: injector and producer coordinates for the eight layouts.
- `pareto_screening_table.csv`: Pareto-screening records used to identify representative layouts.
- `pair_differences.csv`: response differences for the F1, F2, and F3 paired comparisons.
- `response_matched_controls.csv`: matched-layout response controls used in the ensemble analysis.
- `sr3_manifest.csv`: metadata and relative paths for the eight source `.sr3` files.

The CMG `.sr3` binaries are not stored in this GitHub repository. Each file is
approximately 9-10 GB and exceeds GitHub's supported file and repository limits.
The manifest records the corresponding source file without exposing local absolute
paths. The `.sip` files, geological model, property grids, and CMG input files are
also excluded because they are confidential.
