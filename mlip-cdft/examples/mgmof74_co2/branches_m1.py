"""Both branches of the m = 1 CO2 cDFT isotherm and their grand potentials (298 K), for one solid-fluid potential.

feos' desorption isotherm starts the dense branch from the bulk density (capped at the bulk value), which is a gas-like
density here and does not converge. Instead the dense branch is started at the highest pressure from a liquid-like
initial profile rho_0 = min(rho_bulk exp(-beta V_ext), rho_max) and then followed downwards (adaptive steps in ln p).
The dilute branch is followed upwards from low pressure. Where both exist, the equilibrium isotherm switches branch at
equal grand potential (the criterion of feos' equilibrium_isotherm).

Usage: python branches_m1.py <uff|mlip> <out.json>
"""
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

which, out = sys.argv[1], sys.argv[2]
T, NAV, RHO_MAX = 298.0, 6.02214076e23, 20000.0     # rho_max in mol/m3 (liquid CO2 ~ 18-25 kmol/m3)
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
func = pcsaft(["carbon dioxide"], PARS)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
if which == "uff":
    pore = LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T)
else:
    pore = MLIPPore3D(VextResult.load("vext_CO2_MH_production.npz"), T, spacing=0.3, framework_mass_amu=mass, m=1.0)
targets = np.geomspace(1e-4, 40, 41)
newton = feos.DFTSolver().newton(max_iter=100)
robust = (feos.DFTSolver().picard_iteration(log=True, max_iter=400, tol=1e-6, damping_coefficient=0.05)
          .anderson_mixing(max_iter=400, tol=1e-8).newton(max_iter=100))


def state(p):
    return feos.State(func, T * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor")


def record(prof, p):
    return {"p": p, "N": float(prof.total_moles / si.MOL) * NAV, "mmol_g": float(pore.per_mass(prof.total_moles)),
            "omega_J": float(prof.grand_potential / si.JOULE), "max_n3": float(prof.weighted_densities[0][1].max())}


def follow(prof, p_from, ps, label):
    """adaptive continuation in ln p from p_from through the targets ps (monotonic, either direction)"""
    rows, p_cur = [], p_from
    for p_t in ps:
        step, halvings = np.log(p_t / p_cur), 0
        while abs(np.log(p_t / p_cur)) > 1e-12:
            p_new = p_cur * np.exp(step) if abs(step) < abs(np.log(p_t / p_cur)) else p_t
            try:
                prof = pore.initialize(state(p_new), density=prof.density).solve(newton)
                p_cur, step = p_new, step * 1.5
            except Exception:  # noqa: BLE001
                step, halvings = step / 2, halvings + 1
                if halvings > 14:
                    print(f"  {label}: branch ends at {p_cur:.6g} bar", flush=True)
                    return rows, p_cur
        rows.append(record(prof, p_t))
        r = rows[-1]
        print(f"  {label} {p_t:10.4g} bar  {r['mmol_g']:7.3f} mmol/g  N {r['N']:6.3f}  Omega {r['omega_J']:.6e} J  "
              f"max n3 {r['max_n3']:.3f}", flush=True)
    return rows, None


t0 = time.time()
result = {"which": which, "T_K": T}
# dilute branch, upwards
prof = pore.initialize(state(targets[0])).solve(newton)
result["dilute"], result["dilute_end"] = follow(prof, targets[0], targets, "dilute")
# dense branch: liquid-like start at the highest pressure, then downwards
p_hi = targets[-1]
bulk = state(p_hi)
rho_b = float(bulk.density / (si.MOL / si.METER ** 3))
bv = np.where(np.isfinite(pore.vext_k), pore.vext_k / T, np.inf)
rho0 = np.minimum(rho_b * np.exp(-np.minimum(bv, 50.0)), RHO_MAX)[None] * si.MOL / si.METER ** 3
try:
    dense = pore.initialize(bulk, density=rho0).solve(robust)
    print(f"dense start at {p_hi} bar converged ({time.time()-t0:.0f}s)", flush=True)
    result["dense"], result["dense_end"] = follow(dense, p_hi, targets[::-1], "dense")
except Exception as e:  # noqa: BLE001
    print(f"dense start at {p_hi} bar FAILED: {str(e)[:120]}", flush=True)
    result["dense"], result["dense_end"] = [], None
# equilibrium: switch where the grand potentials are equal
d = {r["p"]: r for r in result["dense"]}
both = [(r["p"], r["omega_J"] - d[r["p"]]["omega_J"]) for r in result["dilute"] if r["p"] in d]
cross = [(a[0], b[0]) for a, b in zip(both, both[1:]) if np.sign(a[1]) != np.sign(b[1])]
result["omega_crossing_between_bar"] = cross
print("pressures where both branches exist:", [round(p, 6) for p, _ in both])
print("grand potentials cross between:", cross, f"({time.time()-t0:.0f}s)")
json.dump(result, open(out, "w"), indent=1)
