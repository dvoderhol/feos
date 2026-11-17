"""Final CO2 isotherms in Mg-MOF-74 at 298 K with PC-SAFT cDFT (feos): m = 1 CO2 (parameters/co2_pcsaft_m1.json),
AntiSymWhiteBear FMT (feos' White Bear default fails for the strongly localised molecules at the Mg sites: wrong
zero-dimensional limit + FFT ringing of n2v; see fmt_version_test.py), production uMOF-MH V_ext, UFF baseline.
Writes iso_CO2_final_grid03.json (absolute adsorption; superseded by review_followup.py, which writes the final
iso_CO2_final.json on the nested 0.4 -> 0.2 A grid; the 0.3 A grid here changes K_H by -3.5 %)."""
import json
import sys
import time
import numpy as np
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz")
pores = {"cDFT + MLIP (uMOF-MH)": MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0),
         "cDFT + UFF (standard)": LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T)}
out = {"T_K": T, "fluid": par, "fmt": "AntiSymWhiteBear", "spacing": 0.3, "vext_info": res.info}
for name, pore in pores.items():
    t = time.time()
    iso = adaptive_isotherm(func, pore, T, np.geomspace(1e-4, 40, 61))
    out[name] = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
                 "p_end": iso["p_end"], "K_H": pore.henry_coefficient(T)}
    print(f"{name}: {len(iso['pressure_bar'])} points to {iso['pressure_bar'][-1]:.3g} bar, branch end {iso['p_end']}, "
          f"K_H {out[name]['K_H']:.3e} ({time.time()-t:.0f}s)", flush=True)
json.dump(out, open("iso_CO2_final_grid03.json", "w"), indent=1, default=str)
