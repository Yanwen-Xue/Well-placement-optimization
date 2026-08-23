# Data Availability

The Python source code is provided for transparency and reproducibility of the workflow.

The geological model, reservoir-model input files, petrophysical property grids, and
full-physics simulation outputs used in this study are derived from the Shenhua Ordos CO2
geological storage demonstration project and are subject to confidentiality restrictions.
Therefore, these data are not publicly available.

Public derived data are provided under `public_data/paper_cases_1_8/`. These files
contain the paper-case mapping, selected well coordinates, pressure and plume
metrics, 0--80 year time-series summaries, paired response differences, and the
Pareto-screening table. They do not contain the geological model or the original
simulation binaries.

The repository also provides `public_data/Supplementary_Data_679_Layouts_Three_Responses.csv`
with the well locations, three response metrics, and pressure-peak times for all 679 completed
layouts, and `public_data/Supplementary_Data_Pareto_240_Layouts_Three_Responses.csv` with the
well locations and three surrogate-predicted responses for the 240 non-dominated layouts. These
are derived layout and response tables only and contain no restricted model files.

The eight selected CMG SR3 files are listed in `public_data/paper_cases_1_8/sr3_manifest.csv`
but are not included in the GitHub repository because each file is approximately
9--10 GB. SIP files and CMG input files are also excluded.
