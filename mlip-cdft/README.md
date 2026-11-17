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
