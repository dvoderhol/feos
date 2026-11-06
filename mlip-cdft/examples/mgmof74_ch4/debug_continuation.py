"""Pressure continuation test: fine log-spaced pressure steps with two solver chains."""
import sys
import time

import numpy as np
import si_units as si
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, interpolate_periodic  # noqa: E402

res = VextResult.load("vext_MH_smoke.npz")
T, mass = 298.0, 728.124
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
n = [int(np.ceil(L / 0.3)) for L in res.cellpar[:3]]
bv = interpolate_periodic(res.vext(T) / T, n, cap=50.0)
v = np.where(bv >= 50, np.inf, bv * T)
p = np.geomspace(0.1, 100, 40)
for name, solver in {"newton": feos.DFTSolver().newton(),
                     "logpicard+newton": feos.DFTSolver().picard_iteration(log=True, max_iter=200, tol=1e-6).newton()}.items():
    t = time.time()
    iso = isotherm(func, v, res.cellpar, T, p, mass, solver=solver)
    ok = np.isfinite(iso["mol_per_kg"])
    print(f"{name}: {ok.sum()}/{len(p)} converged, first failure at {p[~ok][0] if (~ok).any() else None} bar, {time.time()-t:.0f}s", flush=True)
    for pb, nk, nm in list(zip(p, iso["mol_per_kg"], iso["molecules_per_cell"]))[::4]:
        print(f"   {pb:8.3f} bar  {nk:7.3f} mol/kg  {nm:6.3f}/cell", flush=True)
