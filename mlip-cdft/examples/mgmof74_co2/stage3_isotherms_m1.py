"""CO2 in Mg-MOF-74 at 298 K with PC-SAFT cDFT, spherical CO2 (m = 1, parameters/co2_pcsaft_m1.json, option B):
  * cDFT + MLIP: uMOF-MH insertion energies, Boltzmann average over 48 linear-molecule directions (MLIPPore3D)
  * cDFT + UFF : UFF LJ 12-6 with the same m = 1 CO2 sigma/epsilon (LJPore3D), standard approach
The adsorption branch is followed in pressure (adaptive_isotherm); where it stops, the isotherm is continued
in the number of molecules (canonical_isotherm, feos PoreSpecification.Moles). Absolute adsorption.

Usage: python stage3_isotherms_m1.py <vext.npz> <fine_spacing_A> <out.json>
"""
import json
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, adaptive_isotherm, canonical_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

npz, spacing, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
T = 298.0
par = json.load(open("/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"))[0]
func = pcsaft(["carbon dioxide"], "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json")
res = VextResult.load(npz)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
pores = {"cDFT + MLIP (uMOF-MH)": MLIPPore3D(res, T, spacing=spacing, framework_mass_amu=mass, m=1.0),
         "cDFT + UFF (standard)": LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=spacing, cap_k=50 * T)}
p = np.geomspace(1e-4, 40, 36)
result = {"T_K": T, "fluid": par, "spacing": spacing, "vext_info": res.info}
for name, pore in pores.items():
    t = time.time()
    iso = adaptive_isotherm(func, pore, T, p)
    entry = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
             "p_end": iso["p_end"], "K_H": pore.henry_coefficient(T), "control": ["pressure"] * len(iso["pressure_bar"])}
    print(f"--- {name}: K_H {entry['K_H']:.3e} mol/(kg Pa); pressure control to {iso['pressure_bar'][-1]:.3g} bar, "
          f"branch end {iso['p_end']} ({time.time()-t:.0f}s)", flush=True)
    if iso["p_end"] is not None:  # continue in N from the last converged state
        n0 = iso["mol_per_kg"][-1] * pore.framework_mass_amu * 1.66053906660e-27 * 6.02214076e23  # molecules/cell
        can = canonical_isotherm(func, pore, T, np.arange(n0 + 0.1, n0 + 40, 0.1), density=iso["profiles"][-1].density)
        keep = can["pressure_bar"] > iso["pressure_bar"][-1]
        entry["pressure_bar"] += can["pressure_bar"][keep].tolist()
        entry["mmol_per_g"] += can["mol_per_kg"][keep].tolist()
        entry["control"] += ["moles"] * int(keep.sum())
        entry["canonical_stop"] = can["failed"]
        print(f"   continued in N to {can['pressure_bar'].max() if len(can['pressure_bar']) else None} bar; "
              f"p(N) monotonic: {bool(np.all(np.diff(can['pressure_bar']) > 0))}; stop: {can['failed']}", flush=True)
    for pb, n in zip(entry["pressure_bar"], entry["mmol_per_g"]):
        print(f"   {pb:10.4g} bar  {n:7.3f} mmol/g", flush=True)
    result[name] = entry
json.dump(result, open(out, "w"), indent=1, default=str)
