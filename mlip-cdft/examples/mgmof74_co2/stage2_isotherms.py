"""Stage 2 (CPU): CO2 in Mg-MOF-74 at 298 K with PC-SAFT cDFT (feos), two solid-fluid potentials on the same grid:
  * MLIP: uMOF-MH insertion energies, Boltzmann-averaged over orientations (MLIPPore3D)
  * standard: UFF LJ 12-6, Lorentz-Berthelot, 14 A cutoff, x m (LJPore3D, as feos' former Pore3D)
PC-SAFT CO2 from gross2001 (non-polar, m = 2.0729), as in the standard approach. Absolute adsorption.

Usage: python stage2_isotherms.py <vext.npz> <fine_spacing_A> <out.json>
"""
import glob
import json
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, adaptive_isotherm, henry_from_vext, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

npz, spacing, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
T = 298.0
func = pcsaft(["carbon dioxide"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
p = np.geomspace(0.001, 40, 30)
M_CO2 = 2.0729
res = VextResult.load(npz)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
pores = {"MLIP uMOF-MH": MLIPPore3D(res, T, spacing=spacing, framework_mass_amu=float(uc.get_masses().sum()), m=M_CO2),
         "UFF-LJ (standard)": LJPore3D(uc, 2.7852, 169.21, m=M_CO2, spacing=spacing, cap_k=50 * T)}
result = {"T_K": T, "spacing": spacing, "vext_info": res.info}
for name, pore in pores.items():
    t = time.time()
    iso = adaptive_isotherm(func, pore, T, p)
    v = pore.vext_k
    # molecule-level Widom K_H of the potential (exact for a rigid molecule) vs the slope feos actually solves
    kh_widom = henry_from_vext(v, pore.cellpar, T, pore.framework_mass_amu)
    slope = iso["mol_per_kg"][0] / (iso["pressure_bar"][0] * 1e5) if len(iso["pressure_bar"]) else float("nan")
    print(f"--- {name}: grid {v.shape}, min V {v[np.isfinite(v)].min():.0f} K, <exp(-V/kT)> K_H {kh_widom:.3e}, "
          f"feos slope at {p[0]} bar {slope:.3e} mol/(kg Pa), branch end {iso['p_end']} ({time.time()-t:.0f}s)", flush=True)
    for pb, nk in zip(iso["pressure_bar"], iso["mol_per_kg"]):
        print(f"   {pb:8.3f} bar  {nk:7.3f} mmol/g", flush=True)
    result[name] = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
                    "p_end": iso["p_end"], "K_H_widom": kh_widom, "feos_slope": slope}

exp = {}
for f in sorted(glob.glob("experiment/298K__*.csv")):
    if "mmol/g" not in open(f).read().splitlines()[2]:
        continue
    d = np.loadtxt(f, delimiter=",", comments="#"); d = d[np.argsort(d[:, 0])]
    if d[:, 0].max() > 200 or d[:, 1].max() > 30:  # inconsistent units in ISODB (Pa labelled bar / not mmol/g)
        continue
    exp[f.split("/")[-1]] = {"at_1bar": float(np.interp(1.0, d[:, 0], d[:, 1])) if d[0, 0] <= 1 <= d[-1, 0] else None,
                             "p_min": float(d[0, 0]), "p_max": float(d[-1, 0])}
result["experiment_298K_mmol_g"] = exp
print("experiment 298 K, mmol/g at 1 bar:", sorted(round(e["at_1bar"], 2) for e in exp.values() if e["at_1bar"]))
json.dump(result, open(out, "w"), indent=1, default=str)
