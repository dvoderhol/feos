"""Follow-up of the independent review (CPU):
(H2) is the steep low-pressure step of cDFT+MLIP caused by PC-SAFT's mean-field attraction between guests?
     -> same MLIP V_ext with a pure hard-sphere functional (FMT only, d = PC-SAFT d(298 K)) at 1e-4..0.1 bar, where the
        bulk is nearly ideal (same fugacity), vs PC-SAFT; Langmuir-like (GCMC-like) means the attraction causes the step.
(M1) helium void fraction / pore volume (feos' pore_volume convention: He sigma 2.64 A, eps/k 10.9 K, 298 K) with UFF.
(M2) final cDFT isotherms (MLIP uMOF-MH, UFF) on the nested 0.2 A grid (2 x the 0.4 A MLIP grid: Henry integral kept).
"""
import json
import sys
import time
import numpy as np
import si_units as si
import feos
from ase.cell import Cell
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.lj import lj_external_potential  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion, grid_points  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz")
n_nested = [2 * k for k in res.n]
out = {}

# (M1) helium pore volume
at = uc.copy(); at.set_cell(Cell.fromcellpar(uc.cell.cellpar()), scale_atoms=True)
v_he = lj_external_potential(grid_points(n_nested).reshape(-1, 3) @ at.cell.array, at, 2.64, 10.9)
phi_he = float(np.mean(np.exp(-np.minimum(v_he, 50 * T) / T)))
vp = phi_he * at.get_volume() * 1e-24 * 6.02214076e23 / mass
out["helium_void_fraction"], out["pore_volume_cm3_g"] = phi_he, vp
print(f"(M1) helium void fraction {phi_he:.4f}, pore volume {vp:.4f} cm3/g", flush=True)

# (H2) PC-SAFT vs hard spheres (same d) in the step region
d = par["sigma"] * (1 - 0.12 * np.exp(-3 * par["epsilon_k"] / T))
pore = MLIPPore3D(res, T, n_grid=n_nested, framework_mass_amu=mass, m=1.0)
p_lo = np.geomspace(1e-4, 0.1, 19)
for name, func in [("PC-SAFT (AntiSymWhiteBear)", pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)),
                   (f"hard spheres only, d = {d:.3f} A (AntiSymWhiteBear)",
                    feos.HelmholtzEnergyFunctional.fmt(np.array([d]), feos.FMTVersion.AntiSymWhiteBear))]:
    t = time.time()
    iso = adaptive_isotherm(func, pore, T, p_lo)
    out[f"step|{name}"] = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist()}
    print(f"(H2) {name}: ({time.time()-t:.0f}s)", flush=True)
    for pb, n in list(zip(iso["pressure_bar"], iso["mol_per_kg"]))[::2]:
        print(f"     {pb:9.4g} bar  {n:7.3f} mmol/g", flush=True)

# (M2) final isotherms on the nested grid
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
pores = {"cDFT + MLIP (uMOF-MH)": pore,
         "cDFT + UFF (standard)": LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, n_grid=n_nested, cap_k=50 * T)}
final = {"T_K": T, "fluid": par, "fmt": "AntiSymWhiteBear", "grid": n_nested, "vext_info": res.info,
         "pore_volume_cm3_g": vp, "helium_void_fraction": phi_he}
for name, pr in pores.items():
    t = time.time()
    iso = adaptive_isotherm(func, pr, T, np.geomspace(1e-4, 40, 61))
    final[name] = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
                   "p_end": iso["p_end"], "K_H": pr.henry_coefficient(T)}
    print(f"(M2) {name}: nested grid {n_nested}, K_H {final[name]['K_H']:.4e}, to {iso['pressure_bar'][-1]:.3g} bar "
          f"(end {iso['p_end']}), {time.time()-t:.0f}s", flush=True)
json.dump(out, open("review_followup.json", "w"), indent=1)
json.dump(final, open("iso_CO2_final.json", "w"), indent=1, default=str)
