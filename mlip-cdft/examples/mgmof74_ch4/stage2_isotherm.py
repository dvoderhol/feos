"""Stage 2 (CPU): interpolate MLIP V_ext onto the fine feos grid and compute the PC-SAFT cDFT isotherm.

Usage: python stage2_isotherm.py <vext.npz> <fine_spacing_A> <T_K> <out_prefix>
"""
import json
import sys
import time

import numpy as np
import si_units as si
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, henry_from_vext, isotherm, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult, interpolate_periodic  # noqa: E402

npz, spacing, T, prefix = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
res = VextResult.load(npz)
info = res.info
mass = info["framework_mass_amu"]
v_coarse = res.vext(T)
n_fine = [int(np.ceil(L / spacing)) for L in res.cellpar[:3]]
cap = 50.0  # beta*V cap used by feos
bv_fine = interpolate_periodic(v_coarse / T, n_fine, cap=cap)
v_fine = np.where(bv_fine >= cap, np.inf, bv_fine * T)
print(f"coarse grid {res.n} -> fine grid {tuple(n_fine)}; min V {np.min(v_coarse):.1f} K (coarse), {np.min(v_fine):.1f} K (fine)")

func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
kh_c = henry_from_vext(v_coarse, res.cellpar, T, mass)
kh_f = henry_from_vext(v_fine, res.cellpar, T, mass)
grid = periodic_grid(res.cellpar, n_fine)
bulk = feos.State(func, T * si.KELVIN, pressure=0.01 * si.BAR)
t0 = time.time()
prof = feos.PoreProfile(grid, bulk, external_potential(v_fine)).solve()
kh_feos = float(prof.henry_coefficients[0] / (si.MOL / si.PASCAL)) / (mass * 1.66053906660e-27)
print(f"K_H [mol/(kg Pa)]: V_ext coarse {kh_c:.4e} | V_ext fine {kh_f:.4e} | feos PoreProfile {kh_feos:.4e}  ({time.time()-t0:.1f} s)")

p = [0.1, 1.0, 5.0, 10.0, 20.0, 35.0, 50.0, 65.0, 100.0]
t0 = time.time()
iso = isotherm(func, v_fine, res.cellpar, T, p, mass)
print(f"isotherm ({len(p)} points) {time.time()-t0:.1f} s")
for pb, nm, nk in zip(iso["pressure_bar"], iso["molecules_per_cell"], iso["mol_per_kg"]):
    print(f"  p = {pb:7.2f} bar   N_abs = {nm:7.3f} molecules/cell   {nk:7.3f} mol/kg")
json.dump({"T_K": T, "n_coarse": list(res.n), "n_fine": n_fine, "K_H_coarse": kh_c, "K_H_fine": kh_f,
           "K_H_feos": kh_feos, "pressure_bar": iso["pressure_bar"].tolist(),
           "molecules_per_cell": iso["molecules_per_cell"].tolist(), "mol_per_kg": iso["mol_per_kg"].tolist(),
           "vext_info": info}, open(prefix + ".json", "w"), indent=1, default=str)
