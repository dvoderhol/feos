# feos_mlip — MLIP (uMOF) external potentials for feos cDFT

Separate Python package in the `mlip-cdft` branch of the feos fork. It leaves the upstream feos
package untouched and only uses its public Python API (`Grid.periodical_3d`, `PoreProfile`,
`Adsorption`). Requires the `umof` conda env (torch, mace-torch[polar], cuequivariance, ase, spglib, feos).

## Idea

Standard PC-SAFT cDFT for MOFs (e.g. feos `pore3d.rs` before #376, Gross group work): the
solid–fluid potential is a Lennard-Jones sum over framework atoms (UFF/DREIDING, Lorentz–Berthelot,
14 Å cutoff, × m). Here it is replaced by insertion energies of the real guest molecule computed
with a machine-learned potential, **energies only**:

    U(r, R)  = E(host + guest(r, R)) − E(host) − E(guest)
    V_ext(r) = −kT ln ⟨exp(−U(r, R)/kT)⟩_R          (Boltzmann orientation average)

`r` is the guest reference atom (the PC-SAFT segment centre, e.g. C of CH4), `R` a rotation.
With this definition ∫exp(−V_ext/kT) dr equals the Widom insertion integral, so the cDFT
Henry coefficient is exact for the MLIP.

## Modules

| module | content |
|---|---|
| `engine.py` | `InsertionEngine`: batched energy-only MACE evaluation; host–host and guest–guest edges built once, guest–host edges on the GPU per batch; no forces, no autograd; U from per-atom energy differences (float32-safe). Verified against the ASE calculator: 0.0000 meV (float64), ≤0.03 meV (uMOF-MH float32), ≤0.4 meV (uMOF-POLAR float32, total-energy differences) |
| `vext.py` | feos grid (`(i+0.5)/n`), symmetry reduction (operations that map the grid onto itself, e.g. inversion after `center_on_inversion`), orientation sets, overlap screening, `VextResult` (raw U kept → V_ext at any T), periodic cubic interpolation coarse → fine grid |
| `cdft.py` | feos glue: `periodic_grid`, `external_potential`, `isotherm`, `henry_from_vext` |

## Conventions and caveats

* V_ext is exact in feos only for m = 1 fluids (feos divides V_ext by m and spreads it along the chain).
* feos treats V_ext as temperature independent in all T-derivatives (enthalpy of adsorption);
  the orientation-averaged V_ext depends on T → enthalpies need the −T ∂V/∂T correction (not yet done).
* The MLIP host must be a supercell in which guest images are further apart than the receptive
  field (2 × r_max = 12 Å for uMOF) plus the guest size (`InsertionEngine` warns otherwise).
* feos returns absolute adsorption; excess needs a pore volume (not yet done).

## API (mirrors feos.Pore1D)

```python
from feos_mlip.cdft import MLIPPore3D, LJPore3D, pcsaft, adaptive_isotherm
pore = MLIPPore3D(VextResult.load("vext.npz"), temperature=298.0, spacing=0.3)   # or LJPore3D(atoms, sigma, eps/k, m)
iso = feos.Adsorption.adsorption_isotherm(func, 298 * si.KELVIN, p, pore.grid, pore.external_potential)
prof = pore.initialize(bulk_state).solve()        # -> feos.PoreProfile (henry_coefficients, enthalpy, ...)
```

## Status (2026-11-05, after independent review)

* Engine verified against ASE (CPU, float64 3e-10 meV; float32 <= 0.2 meV); grid convention, inversion
  symmetry, Boltzmann average, `lj.py` (matches the former feos pore3d.rs, all images within the cutoff).
* Fixed: `interpolate_periodic` now interpolates exp(-beta V) linearly (the former cubic spline of the capped
  beta V created spurious wells; K_H was inflated by 17-59 %). Earlier numbers are superseded.
* `henry_coefficient` is only defined for m = 1 (as in feos). For CO2 (gross2001, m = 2.07) the feos chain
  term smears the molecule-centred MLIP potential: feos slope 3.0e-4 vs <exp(-beta V)> 7.8e-3 mol/(kg Pa).
* The adsorption branch ends (Newton from the previous profile fails; same for 3 cell shapes, grid-converged):
  CH4 UFF-LJ ~9-10.7 bar, CH4 MLIP 3.7 bar, CO2 MLIP 0.65 bar, CO2 UFF-LJ reaches 40 bar. Whether this is a
  spinodal (first-order transition of the cDFT model) is not yet shown -> trace N(mu) with PoreSpecification.Moles.
* feos' `Adsorption::isotherm` restarts from the bulk after a failed point; for increasing pressures this did
  not lose any reachable point in our tests, so no patch for now.
* Open: N(mu) trace, grid/orientation convergence (use orientations("linear") for CO2), -T dV/dT enthalpy
  correction, excess adsorption, m = 1 vs gross2001 for CO2.

## Additions (2026-11-06)

* `parameters/co2_pcsaft_m1.json` (+ `fit_co2_m1.py`): spherical PC-SAFT CO2 (m = 1, sigma 3.654 A, eps/k 250.6 K)
  fitted to the Span-Wagner reference (CoolProp set shipped with feos), 220-295 K; gas fugacity at 298 K within
  1 % up to 40 bar. With m = 1 the molecule-centred, orientation-averaged MLIP potential enters feos exactly.
* `vext.orientations("linear", n)`: Fibonacci directions on a hemisphere for linear centrosymmetric molecules.
* `vext.refine_orientations`: dense orientation sets on the deep wells only (CO2: K_H 9.95e-3 (48) ->
  9.645e-3 (96) -> 9.636e-3 (192) -> 9.632e-3 (384)).
* `vext.apply_overlap_guard`: guest-atom guard without new evaluations. Use 1.8 A: uMOF-MH has extrapolation
  holes (CO2 inside a linker ring at 1.45-1.53 A, U = -4.8 eV); physical binding never comes closer than 2.04 A.
* `cdft.canonical_isotherm`: isotherm traced in N with PoreSpecification.Moles (feos returns the bulk in
  equilibrium -> p(N)). For m = 1 CO2 it shows a van der Waals loop (first-order transition of the cDFT model)
  for both potentials; feos' desorption/equilibrium isotherms do not converge at the dense start; open.

## Root cause of the "branch end" (2026-11-06): White Bear FMT + strongly localised molecules

With molecules localised at strong sites (Mg open metal sites; rho*sigma^3 up to 14-22 on the grid) the FFT-evaluated
FMT weighted densities violate their continuum bounds (n2, n3 < 0; |n2v|/n2 up to ~6000 at thousands of grid points).
feos' default White Bear FMT has no guard against this and the wrong zero-dimensional (single cavity) limit: the
solution develops a spurious van der Waals loop and the solvers stop (CH4+UFF 10.7 bar, CO2+UFF 0.65 bar,
CO2+MLIP 0.0016 bar; independent of grid spacing and cell shape). feos' built-in `FMTVersion.AntiSymWhiteBear`
(correct 0D limit, clamps |n2v| <= n2) removes it: all isotherms continue smoothly to 40-100 bar, dilute-limit results
unchanged. Use `pcsaft(..., fmt_version=feos.FMTVersion.AntiSymWhiteBear)` for adsorption with localised sites.
`examples/mgmof74_co2/final_isotherms.py` produces the final CO2 isotherms this way.

## Independent review 2 (2026-11-06): corrections

* AntiSymWhiteBear removes the solver failure, not the steep low-pressure step of cDFT + MLIP. The step comes from
  PC-SAFT's mean-field attraction between molecules localised at neighbouring Mg sites: with a hard-sphere-only
  functional (same V_ext, same d) the isotherm is Langmuir-like (0.01 bar: 4.3 vs 7.8 mmol/g with PC-SAFT; GCMC with
  the same uMOF-MH potential 4.9). cDFT vs GCMC therefore tests the functional/fluid model, not the potential.
* V_ext interpolation keeps the Henry integral only on nested grids: final cDFT runs use n_feos = 2 x n_MLIP
  (0.4 -> 0.2 A); 0.4 -> 0.3 A had changed K_H by -3.5 %.
* Simulations give absolute adsorption; the experiment is (likely) excess. Helium pore volume (feos convention,
  UFF): void fraction 0.737, 0.81 cm3/g; excess -> absolute adds 0.03 / 0.35 / 1.18 mmol/g at 1 / 10 / 30 bar.
* uMOF-POLAR (own relaxed structure, same protocol): K_H 2.05e-3 vs 9.63e-3 (uMOF-MH) mol/(kg Pa); experimental
  initial slope 2.2e-3 (lower bound). final_isotherms_polar.py.

## Framework flexibility: `feos_mlip/flex.py` (2026-11-17)

Implements the notes "Harmonic approximation for framework flexibility in cDFT": harmonic empty framework (Hessian H
of the MLIP host at R0, translations projected out), linear host-guest coupling f(x) = F_host(host + guest) - F_host(host)
(`InsertionEngine.energies_forces`, one backward pass per batch), framework integrated out:
delta(x) = -1/2 f^T H+ f (one-body, infinite dilution) and w(x, x') = -f^T H+ f' (basin mean field at finite loading,
v_w = -f^T ubar_k, ubar_k = H+ (F_tot - F_k)), solved self-consistently around feos (`FlexibleMLIPPore3D.solve`:
feos solves the PC-SAFT profile for V_eff, the outer loop updates ubar; damped Picard). Optional (A8) cumulant variant.

* `HarmonicFramework`: H+ on the translation-free subspace; checks (A5) (acoustic sum rule, exactly 3 zero modes,
  no negative eigenvalues). Mg-MOF-74 2x1x1 (uMOF-MH, float64): ASR 4e-15, lambda_min 0.059 eV/A^2.
* Symmetry: inversion images reconstructed with the atom permutation, f(Ix)_perm(a) = -f(x)_a; the primitive-cell density
  is replicated into the supercell copies. Verified: v_w equals the numerical functional derivative of F_w (rel. 1e-7..8e-5,
  inverted and direct points).
* Validation of delta against explicit framework relaxation at fixed guest (framework centroid fixed = subspace U):
  ratio 0.99-1.07 at the Mg sites (max harmonic displacement u* < 0.1 A), 0.82-0.88 at u* ~ 0.16 A, but only 0.27-0.66
  for u* > 0.3 A (soft modes; e.g. -0.33 vs -1.25 eV). Unfiltered, delta creates spurious wells (K_H x 1e11).
  Therefore `FlexibleVext(..., u_cut=0.2)`: configurations with max|H+ f| > u_cut are treated rigidly (delta = 0, f = 0);
  this underestimates their (real but smaller) relaxation. K_H(flex)/K_H(rigid) = 2.0 / 3.6 / 4.9 for u_cut 0.1/0.2/0.3 A.
* Finite loading: the collective mean displacement max|<u>| exceeds the harmonic regime (0.56-0.70 A) above ~2e-3 bar;
  results there are flagged invalid.
* Linear coupling can only increase adsorption (delta <= 0); a decrease (stiffening, thermal fluctuation of the sites)
  needs the quadratic coupling (notes, appendix).

## Fluid-model test: cDFT + uMOF-MH with different PC-SAFT CO2 models (2026-11-18)

`examples/mgmof74_co2/fluid_model_test.py <variant>` (same V_ext, grid, FMT and solver as `review_followup.py`):
* `m1q`: m = 1, Q = 4.4 DA, refit to Span-Wagner (`parameters/fit_co2_m1_q.py` -> `co2_pcsaft_m1_q4.4.json`;
  sigma 3.710 A, eps/k 197.4 K; AARD p_sat 4.7 %, rho_liq 1.2 % vs 8.4 % / 2.8 % for the m = 1, Q = 0 set).
  V_ext enters exactly; the isotherm changes by <= 3 % (0.1-40 bar) from the baseline.
* `g2005`: Gross 2005 quadrupolar CO2 (m = 1.513). feos' homosegmented Euler-Lagrange equation
  ln rho = ln f - (dF_res/drho + beta V)/m together with the hard-chain term -(m-1) rho [ln(y lambda) - 1] gives the
  dilute limit m ln g = -beta V + (m-1)[ln(g*w) + (g/(g*w))*w - 1] (w: delta shell of radius d): exact exp(-beta V)
  for potentials slowly varying over d = 3.11 A, but narrow wells (the Mg sites) are smeared. Effective K_H 1.28e-3
  instead of 9.63e-3 mol/(kg Pa) (independent FFT solution of the dilute-limit equation: 1.272e-3). The closer
  agreement with experiment of this variant is therefore mainly a V_ext mapping artefact, not a fluid-fluid effect.
* `g2005mV` (m V passed): INVALID, double counts V (dilute limit exp(-m beta V)); kept only as a record.

## Published NequIP potentials in cDFT (2026-11-19): examples/mgmof74_co2/ext_models

`feos_mlip/engine_nequip.py` (deployed NequIP TorchScript, energy-only path validated against the full model),
`feos_mlip/vext.py: pair_thresholds_vdw, apply_pair_guard` (the papers' pairwise vdW overlap rules, post hoc).
* Goeminne et al., JCTC 19, 6313 (2023), Zenodo 7904959: PBE-D3(BJ)/TS/MBD models on their 54-atom framework
  (2x2x2 host, no symmetry: the framework is P1 within 0.3 A). Trained on intermolecular energies: the correct
  V_ext is E(host+CO2) - E(host) ("paper+Eguest" files), see the note in engine_nequip.py. cDFT K_H 1.99e-3 /
  2.15e-3 / 8.7e-3 mol/(kg Pa) vs the initial slopes of their own GCMC 1.72e-3 / 2.07e-3 / 7.2e-3 (raw uptake series
  from Zenodo, goeminne_gcmc_reference.py). Grid V_ext vs random-insertion Widom (widom_check.py): 3.27e-4 vs
  3.29e-4 +- 0.25e-4 (E_guest-subtracted D3).
* Tayfuroglu & Keskin, JCP 163, 054704 (2025), github otayfuroglu/deepMDMC: fragment-trained (total energies), run on
  OUR uMOF-MH structure (their rigid-GCMC structure is not published): K_H 1.16e-4 (their overlap rule), Widom
  1.08e-4 +- 0.06e-4.
