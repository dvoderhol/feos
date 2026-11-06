# ePCP-SAFT: permittivity parameters of solvents

Fit of the permittivity model of Rueben et al. (2024a) to DDB data with the liquid density of ePCP-SAFT
(identical to PCP-SAFT for systems without ions), for all substances of
[`parameters/epcpsaft/esper2023.json`](../../parameters/epcpsaft/esper2023.json).

| file | content |
| - | - |
| `permittivity.py` | data filter, densities, permittivity from feos (`State.permittivity`), Huber fit, scheme selection, error measures |
| `fit_solvent_permittivity.py` | fit of all substances with DDB data (all filtered data points); writes `results/solvent_permittivity.csv`, with `--write-parameters` the permittivity records in `parameters/epcpsaft/esper2023.json` |
| `iapws.py` | IAPWS-95 (CoolProp) and IAPWS R8-97 static dielectric constant of water |
| `01_water_permittivity.ipynb` | water: fits (scheme P), validation with IAPWS R8-97 |
| `02_solvent_permittivity.ipynb` | all substances: schemes, errors, comparison with the PC-SAFT database of Rueben et al. (2024a) |
| [`parameter_adjustment/`](parameter_adjustment/README.md) | simultaneous fit of the ion parameters of all ions to electrolyte data |

## Licensed data

The DDB export is licensed and must not be committed. `results/`, `*.csv`, `*.png` and `*.pdf` are ignored in
[`.gitignore`](.gitignore); notebooks are committed without outputs and executed into `results/`:

```
export EPCPSAFT_DDB_PERMITTIVITY=<DDB export, tab separated>
export EPCPSAFT_RUEBEN_REPO=<clone of https://gitlab.ethz.ch/epse/molecular-design-public/paper-permittivity-based-on-perturbation-theory>  # optional
python fit_solvent_permittivity.py --write-parameters
jupyter nbconvert --execute --to notebook --output-dir results 01_water_permittivity.ipynb 02_solvent_permittivity.ipynb
```

## Procedure (Rueben et al. 2024a, section 3.1)

- Data: phase "Liquid", frequency < 1000 Hz or not given, all pressures (decision 2026-10-30; Rueben et al. 2024a used
  p < 5 bar or not given, `--max-pressure 5e5`; errors are always reported separately for points below and above 5 bar); points at T >= T_c of the model are
  excluded; the points in `permittivity.EXCLUDED` are removed (MEA, 258.15 and 263.15 K: eps of a solid, decision 2026-10-30).
- Only experimental data (DDB types "Experiment" and "Experimental"; not Smoothed, Calculated, Evaluated, Cited, Unreferenced) and
  no duplicates (`permittivity.duplicates`: same substance, T, p and value from the same reference or from one publication
  under two reference ids, mostly DDB + ThermoML; identical numbers from any reference); decision 2026-11-02. The DDB
  export itself is not modified.
- Substances with fewer than 5 points and no dipole moment (25) are not fitted; piperazine is not in Esper et al. (2023).
- Density: ePCP-SAFT liquid density at the given pressure, else at p = max(1 bar, psat of the model).
- Loss: Huber with delta = 5 % of the mean permittivity for n >= 5, least squares otherwise.
- Schemes: NP (a12alpha), P (a11mu2, a2), All. Water: P (prescribed). n >= 5: NP or P by the lower MAD, then All if its AICc is lower.
  n < 5: P if the dipole moment is >= 1 D, else NP; n = 1 and P: a2 = 0.12.
- Dipole moment for n < 5: DDB export if given, else the scheme of Rueben's `PermittivityParameterDatabasePCSAFT.csv`
  (Rueben et al. 2024a used DIPPR or ORCA).
- Bounds: a11mu2 and a12alpha in [0, 100], a2 in [0, 1].
