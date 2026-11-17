"""Dense branch of the m = 1 CO2 cDFT isotherm at 298 K via continuation in temperature, then equilibrium isotherm.

A liquid-like initial guess at 40 bar does not converge. Instead: solve at T = 500 K, 40 bar (weak adsorption,
converges from the bulk), then cool in steps to 298 K at fixed pressure, each step starting from the previous profile
(the MLIP V_ext is re-averaged at every temperature from the stored insertion energies). At 298 K the dense branch is
followed downwards in pressure; with the dilute branch (branches_<which>.json) the equilibrium isotherm switches
branch at equal grand potential (the criterion of feos' equilibrium_isotherm).

Usage: python branches_tpath.py <uff|mlip>
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

which = sys.argv[1]
T_FINAL, T_START, P_HI, NAV = 298.0, 500.0, 40.0, 6.02214076e23
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
func = pcsaft(["carbon dioxide"], PARS)
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz") if which == "mlip" else None
lj = LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=1e9) if which == "uff" else None
newton = feos.DFTSolver().newton(max_iter=100)


def pore_at(T):
    if which == "uff":
        return lj   # temperature-independent LJ potential (cap applied by feos at 50 kT)
    return MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0)


def state(T, p):
    return feos.State(func, T * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor")


def info(pore, prof):
    return float(pore.per_mass(prof.total_moles)), float(prof.grand_potential / si.JOULE), float(prof.weighted_densities[0][1].max())


t0 = time.time()
pore = pore_at(T_START)
prof = pore.initialize(state(T_START, P_HI)).solve(newton)
print(f"T {T_START} K, {P_HI} bar: {info(pore, prof)[0]:.3f} mmol/g", flush=True)
T, dT = T_START, -20.0
while T > T_FINAL + 1e-9:
    T_new = max(T + dT, T_FINAL)
    try:
        pore = pore_at(T_new)
        prof = pore.initialize(state(T_new, P_HI), density=prof.density).solve(newton)
        T, dT = T_new, max(dT * 1.5, -40.0)
        n, om, n3 = info(pore, prof)
        print(f"T {T:7.2f} K, {P_HI} bar: {n:7.3f} mmol/g  max n3 {n3:.3f}", flush=True)
    except Exception as e:  # noqa: BLE001
        dT /= 2
        if dT > -0.05:
            print(f"temperature path stuck at T = {T:.2f} K ({str(e)[:80]})", flush=True)
            sys.exit(1)

# dense branch at 298 K, downwards in pressure (adaptive ln p steps)
targets = np.geomspace(1e-4, P_HI, 41)[::-1]
rows, p_cur = [], P_HI
for p_t in targets:
    step, halvings = np.log(p_t / p_cur), 0
    while abs(np.log(p_t / p_cur)) > 1e-12:
        p_new = p_cur * np.exp(step) if abs(step) < abs(np.log(p_t / p_cur)) else p_t
        try:
            prof = pore.initialize(state(T_FINAL, p_new), density=prof.density).solve(newton)
            p_cur, step = p_new, step * 1.5
        except Exception:  # noqa: BLE001
            step, halvings = step / 2, halvings + 1
            if halvings > 14:
                break
    if abs(np.log(p_t / p_cur)) > 1e-12:
        print(f"dense branch ends at {p_cur:.6g} bar", flush=True)
        break
    n, om, n3 = info(pore, prof)
    rows.append({"p": float(p_t), "N": float(prof.total_moles / si.MOL) * NAV, "mmol_g": n, "omega_J": om, "max_n3": n3})
    print(f"  dense {p_t:10.4g} bar  {n:7.3f} mmol/g  Omega {om:.6e} J  max n3 {n3:.3f}", flush=True)

out = f"branches_{which}.json"
b = json.load(open(out))
b["dense"], b["dense_method"] = rows, f"temperature continuation {T_START} -> {T_FINAL} K at {P_HI} bar"
d = {round(r["p"], 12): r for r in rows}
both = [(r["p"], r["omega_J"] - d[round(r["p"], 12)]["omega_J"]) for r in b["dilute"] if round(r["p"], 12) in d]
b["omega_crossing_between_bar"] = [(a[0], c[0]) for a, c in zip(both, both[1:]) if np.sign(a[1]) != np.sign(c[1])]
print("both branches at:", [round(p, 6) for p, _ in both], "\nOmega(dilute) - Omega(dense):", [f"{x:.3e}" for _, x in both],
      "\ncrossing between:", b["omega_crossing_between_bar"], f"({time.time()-t0:.0f}s)", flush=True)
json.dump(b, open(out, "w"), indent=1)
