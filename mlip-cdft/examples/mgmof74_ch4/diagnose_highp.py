"""Diagnose the high-pressure cDFT failure on the standard UFF-LJ V_ext (no MLIP involved).

For each converged pressure on a fine adsorption branch: N, dN/dmu, max FMT packing fraction n3,
max rho*d^3 and the V_ext / distance-to-framework where the density peaks. Then the first failing
step is retried with verbose Newton output and smaller pressure increments.
"""
import sys
import time

import numpy as np
import si_units as si
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult  # noqa: E402

T = 298.0
res = VextResult.load("vext_MH_smoke.npz")
v = np.load("v_lj_uff_0.3A.npy")
v = np.where(v < 50 * T, v, np.inf)
n = v.shape
grid = periodic_grid(res.cellpar, n)
pot = external_potential(v)
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
NAV = 6.02214076e23


def report(prof, p):
    rho = np.asarray(prof.density / (si.MOL / si.ANGSTROM ** 3))[0] * NAV  # molecules / A^3
    wd = prof.weighted_densities[0]                                         # FMT: n2, n3, n2v(3)
    n3 = wd[1]
    N = float(prof.total_moles / si.MOL) * NAV
    dndmu = float(np.asarray(prof.dn_dmu / (si.MOL / (si.JOULE / si.MOL))).ravel()[0]) * NAV * 8.314462 * T  # dN/d(beta mu)
    i = np.unravel_index(np.argmax(n3), n3.shape)
    print(f"p {p:8.3f} bar  N {N:7.3f}/cell  dN/d(beta mu) {dndmu:8.3f}  max n3 {n3.max():.4f}  "
          f"max rho*d^3 {rho.max()*3.6**3:6.2f}  at max n3: beta*V {v[i]/T:6.2f}", flush=True)
    return n3.max()


bulk = feos.State(func, T * si.KELVIN, pressure=0.1 * si.BAR)
prof = feos.PoreProfile(grid, bulk, pot).solve(feos.DFTSolver().newton())
p_last, last = 0.1, prof
for p in np.geomspace(0.1, 100, 61)[1:]:
    bulk = feos.State(func, T * si.KELVIN, pressure=p * si.BAR)
    try:
        prof = feos.PoreProfile(grid, bulk, pot, density=last.density).solve(feos.DFTSolver().newton())
        report(prof, p)
        p_last, last = p, prof
    except Exception as e:  # noqa: BLE001
        print(f"p {p:8.3f} bar FAILED from previous profile: {str(e)[:100]}", flush=True)
        break

print(f"\n=== retry beyond {p_last:.3f} bar with tiny steps and verbose Newton ===", flush=True)
for p in p_last * np.array([1.005, 1.01, 1.02, 1.05]):
    bulk = feos.State(func, T * si.KELVIN, pressure=p * si.BAR)
    t = time.time()
    try:
        prof = feos.PoreProfile(grid, bulk, pot, density=last.density).solve(
            feos.DFTSolver(verbosity=feos.Verbosity.Iter).newton(max_iter=100))
        report(prof, p)
        last = prof
    except Exception as e:  # noqa: BLE001
        print(f"p {p:8.3f} bar FAILED ({time.time()-t:.0f}s): {str(e)[:100]}", flush=True)
