"""cDFT CO2 isotherm in Mg-MOF-74 at 298 K with V_ext from uMOF-POLAR (structure relaxed with uMOF-POLAR, production
V_ext from make_production_vext_polar.py), same settings as review_followup.py (m = 1 CO2, AntiSymWhiteBear FMT,
nested feos grid n_feos = 2 x n_MLIP, 0.4 -> 0.2 A). Writes iso_CO2_final_polar.json."""
import json
import sys
import time
import numpy as np
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import MLIPPore3D, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-POLAR.cif"))
res = VextResult.load("vext_CO2_POLAR_production.npz")
pore = MLIPPore3D(res, T, n_grid=[2 * k for k in res.n], framework_mass_amu=float(uc.get_masses().sum()), m=1.0)  # nested grid
t = time.time()
iso = adaptive_isotherm(func, pore, T, np.geomspace(1e-4, 40, 61))
name = "cDFT + MLIP (uMOF-POLAR)"
out = {"T_K": T, "fmt": "AntiSymWhiteBear", "grid": [2 * k for k in res.n], "vext_info": res.info,
       name: {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
              "p_end": iso["p_end"], "K_H": pore.henry_coefficient(T)}}
print(f"{name}: {len(iso['pressure_bar'])} points to {iso['pressure_bar'][-1]:.3g} bar, branch end {iso['p_end']}, "
      f"K_H {out[name]['K_H']:.3e} ({time.time()-t:.0f}s)", flush=True)
json.dump(out, open("iso_CO2_final_polar.json", "w"), indent=1, default=str)
