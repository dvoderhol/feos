"""Empirical check of feos Adsorption::isotherm behaviour (standard UFF-LJ CH4 V_ext computed directly on the grid).
(a) feos adsorption_isotherm with targets [1, 10.5, 2, 5] bar vs (b) the same targets reached by small-step continuation.
If (a) is NaN at 10.5 bar while (b) converges there, a solution exists that feos does not return; the later points
(2, 5 bar) show what feos does after a failure (restart from bulk)."""
import sys
import numpy as np
import si_units as si
import feos
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult  # noqa: E402

T, NAV = 298.0, 6.02214076e23
res = VextResult.load("vext_MH_smoke.npz")
v = np.load("v_lj_uff_0.3A.npy"); v = np.where(v < 50 * T, v, np.inf)
grid, pot = periodic_grid(res.cellpar, v.shape), external_potential(v)
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
targets = np.array([1.0, 10.5, 2.0, 5.0])
iso = feos.Adsorption.adsorption_isotherm(func, T * si.KELVIN, targets * si.BAR, grid, pot)
print("(a) feos adsorption_isotherm:", [f"{p:.1f} bar -> {n:.3f}/cell" for p, n in
      zip(targets, np.asarray(iso.total_adsorption / si.MOL) * NAV)], flush=True)

def solve_from(prof, p):
    return feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=p * si.BAR), pot,
                            density=prof.density).solve(feos.DFTSolver().newton(max_iter=50))

prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=1 * si.BAR), pot).solve()
for p in np.geomspace(1, 10.5, 25)[1:]:
    prof = solve_from(prof, p)
n105 = float(prof.total_moles / si.MOL) * NAV
r = prof.residual if hasattr(prof, "residual") else None
print(f"(b) small-step continuation reaches 10.5 bar: N = {n105:.3f}/cell", flush=True)
try:
    direct = solve_from(feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=1 * si.BAR), pot).solve(), 10.5)
    print(f"(c) one jump 1 -> 10.5 bar from the 1 bar profile: converged N = {float(direct.total_moles/si.MOL)*NAV:.3f}")
except Exception as e:  # noqa: BLE001
    print(f"(c) one jump 1 -> 10.5 bar from the 1 bar profile: FAILED ({str(e)[:70]})")
try:
    bulk = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=10.5 * si.BAR), pot).solve()
    print(f"(d) 10.5 bar started from the bulk density: converged N = {float(bulk.total_moles/si.MOL)*NAV:.3f}")
except Exception as e:  # noqa: BLE001
    print(f"(d) 10.5 bar started from the bulk density: FAILED ({str(e)[:70]})")
