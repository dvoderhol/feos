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
