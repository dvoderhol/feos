"""First-order transition in the cDFT model (m = 1 CO2, 298 K): desorption branch (feos starts it from a liquid-like
density at the highest pressure) and feos' equilibrium isotherm (grand-potential criterion), for both potentials."""
import json
import sys
import time
import numpy as np
import si_units as si
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
par = json.load(open("/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"))[0]
func = pcsaft(["carbon dioxide"], "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json")
res = VextResult.load(sys.argv[1])
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
pores = {"cDFT + UFF (standard)": LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T),
         "cDFT + MLIP (uMOF-MH)": MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0)}
p = np.geomspace(1e-4, 40, 36) * si.BAR
out = {}
for name, pore in pores.items():
    for branch, fn in [("desorption", feos.Adsorption.desorption_isotherm), ("equilibrium", feos.Adsorption.equilibrium_isotherm)]:
        t = time.time()
        try:
            iso = fn(func, T * si.KELVIN, p, pore.grid, pore.external_potential, solver=feos.DFTSolver().newton(max_iter=100))
            n = pore.per_mass(iso.total_adsorption)
            pb = np.asarray(iso.pressure / si.BAR)
            ok = np.isfinite(n)
            print(f"--- {name} {branch}: {ok.sum()}/{len(n)} converged ({time.time()-t:.0f}s)", flush=True)
            for a, b in zip(pb[ok], n[ok]):
                print(f"   {a:10.4g} bar  {b:7.3f} mmol/g", flush=True)
            out[f"{name}|{branch}"] = {"pressure_bar": pb[ok].tolist(), "mmol_per_g": n[ok].tolist()}
        except Exception as e:  # noqa: BLE001
            print(f"--- {name} {branch}: FAILED {type(e).__name__}: {str(e)[:150]} ({time.time()-t:.0f}s)", flush=True)
json.dump(out, open("iso_CO2_m1_branches.json", "w"), indent=1)
