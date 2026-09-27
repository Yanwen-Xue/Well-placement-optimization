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

These are aggregate properties and non-spatial simulation settings. This directory
does not contain the three-dimensional geological model, grid-cell permeability or
porosity fields, well logs, seismic data, or CMG input files.
