"""Is the end of the adsorption branch a real cDFT transition or a grid artifact? (UFF-LJ V_ext, CH4, 298 K)
(1) desorption branch from 100 bar with three solver chains; (2) spinodal pressure vs grid spacing."""
import sys
import time

import numpy as np
import si_units as si
import feos
from ase.cell import Cell
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion, grid_points  # noqa: E402

T, NAV = 298.0, 6.02214076e23
UFF = {"H": (2.886, 0.044), "C": (3.851, 0.105), "O": (3.500, 0.060), "Mg": (3.021, 0.111)}
res = VextResult.load("vext_MH_smoke.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
cell = Cell.fromcellpar(res.cellpar); uc.set_cell(cell, scale_atoms=True)
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")


def lj(n):
    pts = grid_points(n).reshape(-1, 3) @ cell.array
    sig = 0.5 * (np.array([UFF[s][0] * 2 ** (-1 / 6) for s in uc.get_chemical_symbols()]) + 3.7039)
    eps = np.sqrt(np.array([UFF[s][1] * 503.22 for s in uc.get_chemical_symbols()]) * 150.03)
    w = cell.volume / np.linalg.norm(np.cross(cell.array[[1, 2, 0]], cell.array[[2, 0, 1]]), axis=1)
    nr = np.ceil(14.0 / w).astype(int) + 1
    sh = np.array([(i, j, k) for i in range(-nr[0], nr[0] + 1) for j in range(-nr[1], nr[1] + 1)
                   for k in range(-nr[2], nr[2] + 1)]) @ cell.array
    at = (uc.positions[None] + sh[:, None]).reshape(-1, 3); s2 = np.tile(sig, len(sh)) ** 2; e = np.tile(eps, len(sh))
    v = np.empty(len(pts))
    for s in range(0, len(pts), 256):
        r2 = ((pts[s:s + 256, None] - at[None]) ** 2).sum(-1); sr6 = (s2 / r2) ** 3
        v[s:s + 256] = np.where(r2 < 196.0, 4 * e * (sr6 ** 2 - sr6), 0).sum(1)
    v = v.reshape(n)
    return np.where(v < 50 * T, v, np.inf)


def N_of(prof):
    return float(prof.total_moles / si.MOL) * NAV


# (1) desorption branch from 100 bar
n = [int(np.ceil(L / 0.3)) for L in res.cellpar[:3]]
pot = external_potential(lj(n)); grid = periodic_grid(res.cellpar, n)
chains = {"newton": feos.DFTSolver().newton(max_iter=200),
          "picard(log)+newton": feos.DFTSolver().picard_iteration(log=True, max_iter=300, tol=1e-6).newton(max_iter=100),
          "picard(damped 0.02)+anderson+newton": feos.DFTSolver().picard_iteration(max_iter=500, tol=1e-5, damping_coefficient=0.02).anderson_mixing(max_iter=500, tol=1e-8).newton(max_iter=100)}
for name, solver in chains.items():
    t = time.time()
    try:
        prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=100 * si.BAR), pot).solve(solver)
        print(f"100 bar from bulk, {name}: OK N={N_of(prof):.3f}/cell max n3 {prof.weighted_densities[0][1].max():.3f} ({time.time()-t:.0f}s)", flush=True)
        for p in [60, 40, 25, 15, 12, 11, 10.5, 10, 8, 6, 4]:
            try:
                prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=p * si.BAR), pot, density=prof.density).solve(feos.DFTSolver().newton(max_iter=100))
                print(f"   desorption {p:6.1f} bar N={N_of(prof):.3f}/cell max n3 {prof.weighted_densities[0][1].max():.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"   desorption {p:6.1f} bar FAILED: {str(e)[:80]}", flush=True)
                break
        break
    except Exception as e:  # noqa: BLE001
        print(f"100 bar from bulk, {name}: FAILED {str(e)[:90]} ({time.time()-t:.0f}s)", flush=True)

# (2) spinodal pressure vs grid spacing (adaptive continuation from 8 bar)
for spacing in [0.3, 0.2, 0.15]:
    n = [int(np.ceil(L / spacing)) for L in res.cellpar[:3]]
    pot = external_potential(lj(n)); grid = periodic_grid(res.cellpar, n)
    t = time.time()
    prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=1 * si.BAR), pot).solve()
    p, step = 1.0, 0.3
    while step > 1e-5 and p < 100:
        try:
            q = p * np.exp(step)
            prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=q * si.BAR), pot, density=prof.density).solve(feos.DFTSolver().newton(max_iter=50))
            p, step = q, step * 1.5
        except Exception:  # noqa: BLE001
            step /= 2
    print(f"spacing {spacing} A grid {n}: adsorption branch ends at {p:.3f} bar, N={N_of(prof):.3f}/cell, "
          f"max n3 {prof.weighted_densities[0][1].max():.3f} ({time.time()-t:.0f}s)", flush=True)
