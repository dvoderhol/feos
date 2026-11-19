"""Fluid-model test for cDFT + MLIP (uMOF-MH), CO2 in Mg-MOF-74, 298 K (2026-11-18).

Question: does the cDFT + uMOF-MH isotherm get closer to experiment if the PC-SAFT CO2 fluid model is changed?
Everything except the fluid model is identical to the final isotherm in review_followup.py (production V_ext, nested
grid n_feos = 2 x n_MLIP, AntiSymWhiteBear FMT, adaptive adsorption branch, 61 pressures 1e-4..40 bar).

Variants (argv[1]):
  m1q      m = 1, Q = 4.4 DA, refit to Span-Wagner (parameters/co2_pcsaft_m1_q4.4.json). V_ext exact.
  g2005    Gross 2005 quadrupolar CO2 (m 1.5131, sigma 3.1869, eps/k 163.33, Q 4.4), feos' standard V_ext convention:
           the potential passed to feos is the molecule potential V, and feos' homosegmented Euler-Lagrange equation
           ln rho = ln f - (dF_res/drho + beta V)/m (no bond integrals for homosegmented PC-SAFT) gives the dilute
           limit rho ~ exp(-beta V/m), i.e. K_H = <exp(-beta V/m)>, NOT the MLIP K_H.
  g2005mV  same fluid, but m*V passed to feos, so that the dilute limit is exp(-beta V) and K_H equals the MLIP K_H
           exactly. This is a choice of how the molecule-centred potential is mapped on the homosegmented chain; it is
           not standard feos usage.
  (baseline m = 1, Q = 0: iso_CO2_final.json, not recomputed.)
Check: the loading at the lowest pressure is compared with K_H p from the variant's own dilute-limit Boltzmann factor.
Writes fluid_model_<variant>.json.
"""
import json
import sys
import time
import numpy as np
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import MLIPPore3D, _Pore3DBase, adaptive_isotherm, henry_from_vext, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
PAR = "/local/home/dvoderhol/feos/mlip-cdft/parameters"
VARIANTS = {"m1q": (f"{PAR}/co2_pcsaft_m1_q4.4.json", 1.0),
            "g2005": ("/local/home/dvoderhol/feos/parameters/pcsaft/gross2005_literature.json", 1.0),
            "g2005mV": ("/local/home/dvoderhol/feos/parameters/pcsaft/gross2005_literature.json", None)}
variant = sys.argv[1]
pars_file, scale = VARIANTS[variant]
par = [r for r in json.load(open(pars_file)) if r["identifier"].get("cas") == "124-38-9"][0]
m = float(par["m"])
scale = m if scale is None else scale          # factor applied to V before it is passed to feos

uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz")
n_nested = [2 * k for k in res.n]
base = MLIPPore3D(res, T, n_grid=n_nested, framework_mass_amu=mass, m=1.0)   # identical to review_followup.py
pore = base if scale == 1.0 else _Pore3DBase(base.cellpar, base.vext_k * scale, mass, T, m=m)
# dilute limit of feos' homosegmented equation: rho ~ exp(-beta * V_passed / m)
kh_variant = henry_from_vext(pore.vext_k / m, base.cellpar, T, mass)
kh_mlip = henry_from_vext(base.vext_k, base.cellpar, T, mass)
func = pcsaft(["carbon dioxide"], pars_file, feos.FMTVersion.AntiSymWhiteBear)
print(f"{variant}: m {m}, sigma {par['sigma']}, eps/k {par['epsilon_k']}, Q {par.get('q', 0)}; V passed = {scale:.4f} V_MLIP; "
      f"K_H(dilute limit of this variant) {kh_variant:.4e}, K_H(MLIP) {kh_mlip:.4e} mol/(kg Pa)", flush=True)

t = time.time()
iso = adaptive_isotherm(func, pore, T, np.geomspace(1e-4, 40, 61))
wall = time.time() - t
p, n = iso["pressure_bar"], iso["mol_per_kg"]
lin = kh_variant * p[0] * 1e5 * 1e3        # mmol/g at the lowest pressure from K_H
print(f"  {len(p)} points to {p[-1]:.3g} bar, branch end {iso['p_end']}, {wall:.0f} s", flush=True)
print(f"  dilute-limit check at {p[0]:.1e} bar: cDFT {n[0]:.4e} mmol/g, K_H p {lin:.4e} mmol/g (ratio {n[0]/lin:.4f})",
      flush=True)
for q, x in list(zip(p, n))[::5]:
    print(f"   {q:9.4g} bar  {x:7.3f} mmol/g", flush=True)
name = {"m1q": "cDFT + MLIP (uMOF-MH), PC-SAFT m=1 + Q (refit)",
        "g2005": "cDFT + MLIP (uMOF-MH), PC-SAFT Gross 2005, V/m per segment (feos default)",
        "g2005mV": "cDFT + MLIP (uMOF-MH), PC-SAFT Gross 2005, m*V (exact dilute limit)"}[variant]
json.dump({"T_K": T, "fluid": par, "fluid_file": pars_file, "V_scale_passed_to_feos": scale, "fmt": "AntiSymWhiteBear",
           "grid": n_nested, "wall_s": wall, "threads": 8,
           name: {"pressure_bar": p.tolist(), "mmol_per_g": n.tolist(), "p_end": iso["p_end"],
                  "K_H_dilute_limit": kh_variant, "K_H_MLIP": kh_mlip, "dilute_check_ratio": float(n[0] / lin)}},
          open(f"fluid_model_{variant}.json", "w"), indent=1, default=str)
