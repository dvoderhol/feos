"""First finite-loading test of the flexible cDFT (basin mean field around feos), u_cut 0.2 A, E_c 10 kT, nested grid,
PC-SAFT m = 1 CO2, AntiSymWhiteBear, 298 K. Pressure ladder 1e-4 -> 1 bar (continuation in density and ubar).
Compares: rigid (production V), flexible infinite-dilution V (delta only), flexible with mean field (delta + v_w).
Writes test_flex_isotherm.json."""
import json
import sys
import time

import numpy as np
import feos
import si_units as si
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import MLIPPore3D, pcsaft  # noqa: E402
from feos_mlip.flex import FlexCoupling, FlexibleMLIPPore3D, FlexibleVext, HarmonicFramework  # noqa: E402
from feos_mlip.vext import KB_EV, VextResult, center_on_inversion  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat((2, 1, 1)); host.wrap()
fw = HarmonicFramework(np.load("hessian_2x1x1.npz")["H"])
cp = FlexCoupling.load("coupling_CO2_MH_20kT.npz")
mass = res.info["framework_mass_amu"]
t0 = time.time()
fv = FlexibleVext(res, cp, fw, uc, host, T, e_cut=10 * KB_EV * T, u_cut=0.2)
pore_f = FlexibleMLIPPore3D(fv, framework_mass_amu=mass)
print(f"setup {time.time() - t0:.1f} s; basins {pore_f.n_basins}; K_H flex {pore_f.henry_coefficient():.4e}", flush=True)
pore_r = MLIPPore3D(res, T, n_grid=pore_f.n_fine.tolist(), framework_mass_amu=mass, m=1.0)
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
solver = feos.DFTSolver().newton(max_iter=100)
out = dict(K_H_rigid=pore_r.henry_coefficient(T), K_H_flex=pore_f.henry_coefficient(), n_basins=int(pore_f.n_basins), points=[])
dens = dict(r=None, d=None, f=None)
ubar = None
for p in np.geomspace(1e-4, 1.0, 13):
    bulk = feos.State(func, T * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor")
    rec = dict(p=float(p))
    t = time.time(); pr = pore_r.initialize(bulk, density=dens["r"]).solve(solver); dens["r"] = pr.density
    rec["rigid"] = float(pore_r.per_mass(pr.total_moles)) ; rec["t_rigid"] = round(time.time() - t, 1)
    t = time.time(); pdl = feos.PoreProfile(pore_f.grid, bulk, pore_f.external_potential, density=dens["d"]).solve(solver)
    dens["d"] = pdl.density
    rec["delta_only"] = float(pore_f.per_mass(pdl.total_moles)); rec["t_delta"] = round(time.time() - t, 1)
    t = time.time()
    sol = pore_f.solve(bulk, density=dens["f"], ubar=ubar, solver=solver, mixing=0.5, tol_u=1e-5, verbose=False)
    dens["f"], ubar = sol["profile"].density, sol["ubar"]
    rec["mean_field"] = float(pore_f.per_mass(sol["profile"].total_moles)); rec["t_mf"] = round(time.time() - t, 1)
    rec["iterations"] = sol["iterations"]
    rec["max_u_mean_A"] = float(np.linalg.norm(sol["u_mean"], axis=1).max())
    out["points"].append(rec)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    json.dump(out, open("test_flex_isotherm.json", "w"), indent=1)
