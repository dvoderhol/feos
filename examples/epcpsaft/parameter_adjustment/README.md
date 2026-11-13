# ePCP-SAFT: fitting ion parameters

Simultaneous fit of the parameters of all ions to electrolyte data (all salts and mixtures at once), with exact
derivatives from feos (automatic differentiation through the density iteration and, for the MSA, through the
screening parameter).

| file | content |
| - | - |
| `electrolyte_data.py` | reading the data (CSV format of `~/ePCP_dummy_data`), reduction of every data point to a target quantity and a residual definition with pure-water reference data (IAPWS, ice) only |
| `electrolyte_fit.py` | parameter vector, model (targets and Jacobian via `feos.Property.electrolyte_properties_derivatives`), Huber objective, problem setup, least squares, multistart, continuation, classification, CMA-ES, symmetries and gauges, identifiability, output in feos JSON format |
| `test_electrolyte_fit.py` | tests (exactness, derivatives for Debye-Hückel and MSA, symmetries, parameter recovery, optimizers, smoke test) |
| `adjust_ion_parameters.ipynb` | example: fit to a data folder (committed without outputs) |

Data are not part of the repository: set `EPCPSAFT_ELECTROLYTE_DATA` to the data folder (fit) and `EPCPSAFT_DUMMY_DATA`
to `~/ePCP_dummy_data` (tests). Results are written to `EPCPSAFT_ION_OUTPUT` (default `results/ion_parameters`;
`results/` is ignored by git).

```
EPCPSAFT_DUMMY_DATA=~/ePCP_dummy_data pytest test_electrolyte_fit.py      # ~35 min (extended dummy data)
```
feos evaluates the data points in parallel on all cores (rayon); on shared machines limit the threads, e.g.
`RAYON_NUM_THREADS=4`.

## Model and parameters

ePCP-SAFT with fixed PCP-SAFT water (`parameters/epcpsaft/esper2023.json`, incl. its permittivity record); ion-ion term
Debye-Hückel (`variant="dh"`) or mean spherical approximation (`variant="msa"`, Blum's nonrestricted MSA; packing term Δ
over the ions only, `msa_packing="ions"` (primitive model), or over all species, `msa_packing="total"` (excluded volume
of the solvent in Δ only, as Maribo-Mogensen et al. 2012; Ω and the equation for Γ keep the sums over the ions)).
The Born diameter is the ion diameter 0.88σ plus `born_delta_cation` / `born_delta_anion`.
Adjusted per ion: `sigma`, `epsilon_k`, k_ij(water, ion), `dipole_scaling` (a11μ²), `correlation_integral_parameter`
(a2); global: `born_delta_cation`, `born_delta_anion`. Fixed: `polarizability_scaling` = 0, cation-anion k_ij = 0
(like-charged ions: 1 in feos). Ion permittivity parameters (`permittivity_variant`): "PT2" a11μ² and a2 per ion
(Rueben et al. 2024); "PT1" a11μ² per ion and a2 of every ion equal to a2 of water (not adjusted).
Bounds (`BOUNDS`) only where physics requires them: σ > 0, ε/k ≥ 0, ion-water dispersion energy ≥ 0 (k_iw ≤ 1; in the
cross-energy variables s ≥ 0, u ≥ 0). The Sobol box `SAMPLING` only places starting points.

Apparent molar heat capacities (`Model(cp_model=...)`): residual part of the model plus
- `"ideal_gas"` (default): the ideal-gas heat capacities of the ions, `ideal_gas_cp`: 5/2 R (translation) plus
  rotation and harmonic vibrations of polyatomic ions (fundamentals in aqueous solution, Nakamoto 2009); electronic
  contributions of the free ions from the NIST levels (`electronic_cp`, `cp_electronic=True`; not used by default);
- `"offset"`: one adjusted offset per salt system, composition and temperature (parameters `Cp_offset`);
- `"residual"`: nothing (residual convention of the dummy data).

Reaction equilibria (`Model(reactions=REACTIONS)`): HSO₄⁻ = H⁺ + SO₄²⁻ with ln K(T) of Knopf et al. (2003, Table 2,
eq. 16; molality scale), solved in feos (`ElectrolyteProperties::with_reactions`, Newton for the extents with exact
derivatives with respect to the parameters). A reaction takes part in a system that contains all its reactants or all
its products (`system_species`; HSO₄⁻ is then an ion of the parameter set, `model_ions`); the compositions of the data
are stoichiometric (H2SO4: 2 H⁺ + SO₄²⁻; NaHSO4: Na⁺ + HSO₄⁻, passed to feos as Na⁺ + H⁺ + SO₄²⁻). Targets: osmotic
coefficient with the stoichiometric ion molalities; mean ionic activity coefficient γ±^ν = Π_i (m_i γ_i / m_i^st)^ν_i
(true molalities and activity coefficients of the equilibrium, stoichiometric molalities m_i^st); enthalpies of
dilution from the relative enthalpy L (feos `relative_enthalpy`, including Σ ξ_r ΔH°_r); heat capacities from the
equilibrium heat capacity (central difference in T, feos) plus the ideal-gas heat capacities of the stoichiometric
amounts.

### Exact symmetries

No data can determine these directions. Both act separately on every group of ions that are connected through the
systems of the data; `gauge_fixed` fixes a11μ², a2 and ε/k of one reference ion per group (Cl⁻ where present, otherwise
the anion with most data points). For the full data specification all ions form one group.

- Permittivity gauge: a_i → a_i + c z_i (a11μ² or a2) leaves every electroneutral solution unchanged and shifts every
  single-ion ln γ_i by z_i Ψ (the unmeasurable single-ion activity). After the fit, `macinnes_gauge` shifts a11μ² such
  that γ(K⁺) ≈ γ(Cl⁻) in KCl(aq) (MacInnes convention, least squares at 298.15 K; a2 keeps the value of Cl⁻).
- Dispersion symmetry: ions interact by dispersion only with water and with oppositely charged ions, and only
  ε_ij = √(ε_i ε_j)(1 − k_ij) enters the model. √ε_i → λ√ε_i and (1 − k_iw) → (1 − k_iw)/λ for all cations, √ε_i → √ε_i/λ
  and (1 − k_iw) → λ(1 − k_iw) for all anions leaves every ε_ij unchanged (`dispersion_symmetry`). ε/k of Cl⁻ is set to
  170 K (Held et al. 2014) by this symmetry (`reference_gauge`, no prediction changes), so that all ion ε/k are on the
  usual ePC-SAFT scale.

## Data reduction and residuals (relative deviations of the measured quantity, as for PCP-SAFT)

| property | model quantity | residual |
| - | - | - |
| osmotic coefficient | φ | relative |
| water activity, vapor pressure, freezing / boiling temperature | ln a_w at the temperature of the measurement | relative deviation of a_w, of the vapor pressure, of the freezing-point depression or of the boiling-point elevation, linearized in ln a_w with factors from IAPWS-95 (pure-water vapor, fugacities) and the melting properties of ice |
| mean ionic activity coefficient (molality scale) | γ± | relative |
| density | density (V_φ of the model + IAPWS-95 density of pure water) | relative |
| apparent molar volume, apparent molar heat capacity, enthalpy of dilution | unchanged | divided by the root mean square of the block (can be zero or change sign) |

The reference state of the ions is infinite dilution in pure water, evaluated with ion amounts exactly zero.
Assumptions: pure-water vapor (not valid for volatile solutes such as HCl, HNO3, NH3); freezing points relative to
T = 273.15 K (real data should give the depression relative to the source's own pure-water value).

## Objective (as for PCP-SAFT, Esper et al. 2023)

f_obj = Σ_b w_b / N_b Σ_i L(r_i) over blocks b = (system, property), with the Huber loss L (δ = 0.05):
L = r²/2 for |r| ≤ δ, δ(|r| − δ/2) otherwise. Residuals up to 5 % count quadratically, larger ones linearly, so that
outliers have less influence. `scipy.optimize.least_squares` receives f_i = √(w_b/N_b) ψ(r_i) with
ψ(r) = r (|r| ≤ δ), sign(r)√(2δ|r| − δ²) otherwise, so that ½ Σ f_i² = f_obj, and the exact Jacobian. Weights w_b: 1
(configurable per property). Points that do not converge give NaN residuals, so that the trust-region step is rejected.
(At high ion concentrations the density iteration from the default liquid density can fail although the liquid exists;
`ElectrolyteProperties` then restarts it from the density of the pure solvent scaled by n_total/n_solvent and from 0.75,
0.5, 0.35 and 0.25 of that density. At extreme states (e.g. MgSO4 12 mol/kg, 473 K) the rounding noise of the pressure
can exceed the tolerance of the density iteration of feos, so that single points may not converge.)

## Optimization

1. Problem setup that removes local minima and curved valleys before an optimizer sees them: exact symmetries removed;
   optimizer variables s_i = √ε_i and u_i = √ε_i (1 − k_iw) instead of (ε_i, k_iw) (`Problem(cross_energies=True)`),
   since the ion-water dispersion energy depends only on u_i and the cation-anion energies only on s_i; continuation
   from easy to all data (`continuation`, e.g. first m ≤ 1 mol/kg and 273-323 K).
2. Gradient-based nonlinear least squares with robust loss: trust region reflective (`scipy.optimize.least_squares`,
   bounds, exact Jacobian), as for PCP-SAFT.
3. Many starting points (`multistart`): given starts (e.g. literature values) and a scrambled Sobol sample of the
   parameter box (`SAMPLING`); every start runs for a few evaluations, the best are converged. `waterfall` gives the
   sorted final costs; `classify` decides from the converged starts: "unimodal" (same cost, same parameters),
   "flat" (same cost, different parameters: directions the data do not determine) or "multimodal" (different costs).
4. Fallback for true multimodality: CMA-ES (`cma_es`, pycma) on the same objective, polished by least squares
   (`robust_fit` combines 3 and 4).

Multistart of gradient-based local optimizers from space-filling starts is the established approach for large
parameter estimation problems with exact derivatives (Raue et al. 2013; Villaverde et al. 2019, where a hybrid of
scatter search and a local optimizer was somewhat better).

`block_metrics` gives absolute and relative deviations per block in the measured quantity (a_w, p, freezing-point
depression and boiling-point elevation in K for the colligative data, to first order in the residual).

`classify` re-polishes the converged starts with tight tolerances (with `problem`) and calls the problem multimodal only
if a start ends with a cost higher by more than 1 % and different parameters (more than 0.01 in units of `TYPICAL`).

`identifiability` gives the singular values of the scaled Jacobian, the parameter combinations of the smallest ones and
rough standard errors (pseudo-inverse of the Jacobian with columns scaled to unit norm); with `problem` in the optimizer
variables (s = sqrt(epsilon_k), u = sqrt(epsilon_k) (1 - k_w)).

## Output

`write_parameters` writes the ion records (`<name>.json`), the water-ion k_ij (`<name>_binary.json`) in feos format and
the Born corrections and C_p offsets (`<name>_options.json`).
