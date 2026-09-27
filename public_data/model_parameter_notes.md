# Non-spatial model parameter summary

`model_interval_summary.csv` reproduces the five interval-level statistics reported in
Table 1 of the manuscript. Permeability and porosity are reported as mean, minimum,
and maximum of the current simulation-grid properties; the remaining columns are
interval means. CO2 viscosity is evaluated from initial pressure and temperature.

The GEM model uses equal I- and J-direction permeability and sets K-direction
permeability to 0.1 times I-direction permeability. The outer flow boundaries are
no-flow. The compositional model uses the Peng-Robinson equation of state, the
Henry-law CO2 dissolution option, 0.34 molal salinity, and rock compressibility
of 6.0e-7 kPa^-1 at a 20 MPa reference pressure. Tabulated capillary pressure is zero.

`gem_drainage_relative_permeability.csv` transcribes the drainage water and gas
relative-permeability saturation tables from the final 268 x 203 x 200 GEM input.
The two relative-permeability columns use separate water and gas saturation axes,
which share the same tabulated values in this input. Both capillary-pressure columns
are zero at all tabulated saturations. The file contains no spatial reservoir data.

The Shenhua shallow-overburden density (2500 kg/m3) and caprock Poisson ratio
(0.29) used in the fracture-pressure estimate follow Zhu et al. (2015), Table 1
(doi:10.1016/j.ijggc.2015.08.017). The hydrostatic screening calculation uses
the original model's reference pressure (22.929 MPa), reference depth
(987.241211 m), and constant brine density (1015 kg/m3). The 0.90 multiplier
on estimated fracture pressure is the study's engineering BHP margin.

These are aggregate properties and non-spatial simulation settings. This directory
does not contain the three-dimensional geological model, grid-cell permeability or
porosity fields, well logs, seismic data, or CMG input files.
