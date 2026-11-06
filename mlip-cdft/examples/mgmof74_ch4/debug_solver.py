"""Why does the isotherm fail above 1 bar? Solve single profiles at 5 bar with different solver chains."""
import sys
import time

import numpy as np
import si_units as si
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult, interpolate_periodic  # noqa: E402

res = VextResult.load("vext_MH_smoke.npz")
T = 298.0
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
for spacing in [0.3, 0.2]:
    n = [int(np.ceil(L / spacing)) for L in res.cellpar[:3]]
    bv = interpolate_periodic(res.vext(T) / T, n, cap=50.0)
    pot = external_potential(np.where(bv >= 50, np.inf, bv * T))
    grid = periodic_grid(res.cellpar, n)
    for p in [1.0, 5.0]:
        bulk = feos.State(func, T * si.KELVIN, pressure=p * si.BAR)
        solvers = {
            "newton (default)": feos.DFTSolver(verbosity=feos.Verbosity.Iter).newton(),
            "picard+anderson+newton": feos.DFTSolver(verbosity=feos.Verbosity.Iter).picard_iteration(max_iter=50, tol=1e-5, damping_coefficient=0.05).anderson_mixing(max_iter=200, tol=1e-8).newton(),
        }
        for name, solver in solvers.items():
            t = time.time()
            try:
                prof = feos.PoreProfile(grid, bulk, pot).solve(solver)
                n_ads = float(np.sum(prof.moles / si.MOL)) * 6.02214076e23
                print(f"spacing {spacing} n {n} p {p} bar {name}: OK N={n_ads:.3f}/cell max rho*sigma^3="
                      f"{float(np.max(prof.density/(si.MOL/si.METER**3)))*6.02214076e23*(3.7039e-10)**3:.2f} ({time.time()-t:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"spacing {spacing} n {n} p {p} bar {name}: FAILED {type(e).__name__}: {str(e)[:200]} ({time.time()-t:.0f}s)", flush=True)
