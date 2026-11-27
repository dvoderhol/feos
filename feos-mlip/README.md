# feos-mlip: MLIP external potentials for feos cDFT in periodic frameworks

Python package next to `py-feos` in this fork. It builds the solid–fluid potential of a periodic framework from
insertion energies of the guest molecule computed with a machine-learned interatomic potential (MLIP) and passes it to
feos (`Grid.periodical_3d`, `PoreProfile`, `Adsorption`). It uses only the public feos Python API.

```
pip install -e feos-mlip              # plus the extra of the MLIP used: [mace] or [nequip]
```

## Model

Standard PC-SAFT cDFT for MOFs uses a Lennard-Jones sum over framework atoms (UFF/DREIDING, Lorentz–Berthelot,
cutoff, × m), as in feos' former `pore3d.rs` (removed in #376). Here the potential is computed from the MLIP:

    U(r, R)  = E(host + guest(r, R)) − E(host) − E(guest)
    V_ext(r) = −kT ln ⟨exp(−U(r, R)/kT)⟩_R          (Boltzmann orientation average)

`r` is the guest reference atom (PC-SAFT segment centre), `R` a rotation. With this definition ∫exp(−V_ext/kT) dr is
the Widom insertion integral, so the cDFT Henry coefficient equals that of the MLIP.

Options beyond the rigid framework:
* framework flexibility (`flex.py`): harmonic framework integrated out; one-body relaxation free energy
  δ(x) = −½ fᵀH⁺f in V_ext, and at finite loading a basin mean field solved self-consistently around feos;
* framework-mediated guest–guest interaction (`basins.py`): static pair interaction W_kl per basin pair, as a basin
  mean field ½ Σ_kl W_kl θ_k θ_l (θ_k: guests in basin k), i.e. a `feos.RegionMeanField` with the basins as regions,
  passed to `feos.PoreProfile` / `feos.Adsorption` (`mean_field=`) and solved by feos' Newton solver.

## Modules

| module | content |
|---|---|
| `cdft.py` | feos glue: `periodic_grid`, `external_potential`, `pcsaft`, `henry_from_vext`, `MLIPPore3D`, `LJPore3D`, `adaptive_isotherm`, `canonical_isotherm` |
| `vext.py` | feos grid convention (`(i+0.5)/n`), symmetry reduction, orientation sets, overlap guards, `VextResult` (raw U kept, V_ext at any T), `compute_vext`, `refine_orientations`, periodic interpolation coarse → fine grid |
| `engine.py` | `InsertionEngine`: batched, energy-only MACE evaluation (host–host edges once, guest–host edges per batch on the GPU) |
| `engine_nequip.py` | `NequIPInsertionEngine`, `NequIPCalculator`: deployed NequIP TorchScript models, same interface |
| `lj.py` | `lj_external_potential`: LJ 12-6 baseline (feos' former `pore3d.rs` model, all atom images within the cutoff) |
| `flex.py` | `HarmonicFramework`, `FlexCoupling`, `FlexibleVext`, `FlexibleMLIPPore3D` |
| `basins.py` | `BasinConfigs`, `basin_pair_interaction`, `BasinPairField` (`.mean_field` → `feos.RegionMeanField`) |

## Conventions

* feos divides V_ext by the segment number m, so a molecule-centred V_ext is used exactly only for m = 1 fluids.
* feos treats V_ext as temperature independent in its temperature derivatives (enthalpy of adsorption); the
  orientation-averaged V_ext depends on T.
* The MLIP host must be a supercell in which guest images are further apart than the receptive field plus the guest
  size (`InsertionEngine` warns otherwise).
* feos returns absolute adsorption.
* For fluids with localised molecules at strong sites use `feos.FMTVersion.AntiSymWhiteBear`.
