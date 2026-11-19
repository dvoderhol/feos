"""Flexible-framework cDFT isotherm of CO2 in Mg-MOF-74 (uMOF-MH, 298 K): harmonic framework (notes), validity cut
u_cut = 0.2 A (configurations beyond: rigid), E_c = 10 kT, basin mean field self-consistent around feos (PC-SAFT m = 1,
AntiSymWhiteBear, nested grid 2 x MLIP grid), 61 pressures 1e-4 .. 40 bar (as the rigid isotherms).
For every pressure: loading with the mean field, loading with delta only (infinite-dilution correction), converged
max mean framework displacement max|<u>| (A) and timing. Points with max|<u>| > u_cut lie outside the harmonic regime
validated by explicit relaxations and are flagged 'valid': False. K_H for u_cut 0.1 / 0.2 / 0.3 A.
Writes iso_CO2_flex.json (key 'cDFT + MLIP (uMOF-MH, flexible)': only the valid points)."""
import json
import sys
import time

import numpy as np
import feos
import si_units as si
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import pcsaft  # noqa: E402
from feos_mlip.flex import FlexCoupling, FlexibleMLIPPore3D, FlexibleVext, HarmonicFramework  # noqa: E402
from feos_mlip.vext import KB_EV, VextResult, center_on_inversion  # noqa: E402

T, U_CUT, E_C = 298.0, 0.2, 10 * KB_EV * 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat((2, 1, 1)); host.wrap()
fw = HarmonicFramework(np.load("hessian_2x1x1.npz")["H"])
cp = FlexCoupling.load("coupling_CO2_MH_20kT.npz")
mass = res.info["framework_mass_amu"]
kh = {}
for u in (0.1, 0.2, 0.3):
    kh[u] = FlexibleMLIPPore3D(FlexibleVext(res, cp, fw, uc, host, T, e_cut=E_C, u_cut=u), framework_mass_amu=mass).henry_coefficient()
fv = FlexibleVext(res, cp, fw, uc, host, T, e_cut=E_C, u_cut=U_CUT)
pore = FlexibleMLIPPore3D(fv, framework_mass_amu=mass)
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
solver = feos.DFTSolver().newton(max_iter=100)
pts, dens_d, dens_f, ubar = [], None, None, None
t_all = time.time()
for p in np.geomspace(1e-4, 40, 61):
    bulk = feos.State(func, T * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor")
    rec = dict(p=float(p))
    try:
        t = time.time()
        pd = feos.PoreProfile(pore.grid, bulk, pore.external_potential, density=dens_d).solve(solver)
        dens_d = pd.density
        rec.update(delta_only=float(pore.per_mass(pd.total_moles)), t_delta=round(time.time() - t, 1))
        t = time.time()
        sol = pore.solve(bulk, density=dens_f, ubar=ubar, solver=solver, mixing=0.5, tol_u=1e-5, max_iter=300)
        dens_f, ubar = sol["profile"].density, sol["ubar"]
        umax = float(np.linalg.norm(sol["u_mean"], axis=1).max())
        rec.update(mean_field=float(pore.per_mass(sol["profile"].total_moles)), iterations=sol["iterations"],
                   max_u_mean_A=umax, valid=bool(umax <= U_CUT), t_mf=round(time.time() - t, 1))
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:200]
    pts.append(rec)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    ok = [q for q in pts if q.get("valid")]
    out = {"cDFT + MLIP (uMOF-MH, flexible)": dict(
               pressure_bar=[q["p"] for q in ok], mmol_per_g=[q["mean_field"] for q in ok], K_H=kh[U_CUT],
               K_H_range=[kh[0.1], kh[0.3]], u_cut=U_CUT, e_cut_kT=10,
               note="only points with max|<u>| <= u_cut (harmonic regime validated by explicit relaxations)"),
           "all_points": pts, "K_H_by_u_cut": {str(k): v for k, v in kh.items()},
           "wall_s": round(time.time() - t_all, 1), "cpu_threads": 16}
    json.dump(out, open("iso_CO2_flex.json", "w"), indent=1)
