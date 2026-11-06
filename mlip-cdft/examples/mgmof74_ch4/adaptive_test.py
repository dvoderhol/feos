"""Adaptive pressure continuation (step halving on failure) up to 100 bar, for the UFF-LJ and MLIP V_ext.
Monitors max packing fraction n3 and dN/d(beta mu) to tell a stiff saturating site from a phase transition."""
import sys
import time

import numpy as np
import si_units as si
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult, interpolate_periodic  # noqa: E402

T, NAV = 298.0, 6.02214076e23
res = VextResult.load("vext_MH_smoke.npz")
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
n = (23, 51, 51)
v_lj = np.load("v_lj_uff_0.3A.npy"); v_lj = np.where(v_lj < 50 * T, v_lj, np.inf)
bv = interpolate_periodic(res.vext(T) / T, n, cap=50.0); v_ml = np.where(bv >= 50, np.inf, bv * T)
grid = periodic_grid(res.cellpar, n)
targets = np.geomspace(0.1, 100, 16)

for name, v in [("UFF-LJ", v_lj), ("MLIP uMOF-MH", v_ml)]:
    pot = external_potential(v)
    t0 = time.time()
    prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=targets[0] * si.BAR), pot).solve()
    p_cur, n_sub, n_fail = targets[0], 0, 0
    print(f"--- {name}", flush=True)
    for p_t in targets:
        lnstep = np.log(p_t / p_cur)
        while p_cur < p_t * (1 - 1e-12):
            p_new = min(p_t, p_cur * np.exp(lnstep))
            try:
                prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=p_new * si.BAR), pot,
                                        density=prof.density).solve(feos.DFTSolver().newton(max_iter=50))
                p_cur, n_sub = p_new, n_sub + 1
                lnstep *= 1.5
            except Exception:  # noqa: BLE001
                n_fail += 1
                lnstep /= 2
                if lnstep < 1e-6:
                    break
        if p_cur < p_t * (1 - 1e-12):
            print(f"  stuck at {p_cur:.4f} bar (step < 1e-6 in ln p)", flush=True)
            break
        N = float(prof.total_moles / si.MOL) * NAV
        dndmu = float(np.asarray(prof.dn_dmu / (si.MOL / (si.JOULE / si.MOL))).ravel()[0]) * NAV * 8.314462 * T
        n3 = prof.weighted_densities[0][1].max()
        print(f"  p {p_t:8.3f} bar  N {N:7.3f}/cell  dN/d(beta mu) {dndmu:8.3f}  max n3 {n3:.4f}  "
              f"[{n_sub} solves, {n_fail} failures so far, {time.time()-t0:.0f}s]", flush=True)
