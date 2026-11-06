"""Is the ~9 bar end of the CH4 adsorption branch caused by the oblique cell? Same framework, same UFF-LJ
model, three cell shapes: oblique primitive (54 atoms), hexagonal (162, gamma=120), orthorhombic (324, all 90).
Adaptive pressure continuation 1 -> 100 bar at 0.3 A grid spacing; loading reported per primitive cell."""
import sys
import time

import numpy as np
import si_units as si
import feos
from ase.build import make_supercell
from ase.cell import Cell
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import external_potential, pcsaft, periodic_grid  # noqa: E402
from feos_mlip.lj import lj_external_potential  # noqa: E402
from feos_mlip.vext import grid_points  # noqa: E402

T, NAV, SPACING = 298.0, 6.02214076e23, 0.3
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
hexa = read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_hex_uMOF-MH.cif")
cells = {"orthorhombic (324 atoms)": make_supercell(hexa, [[1, 0, 0], [1, 2, 0], [0, 0, 1]]),
         "hexagonal (162 atoms)": hexa,
         "oblique primitive (54 atoms)": read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif")}
for name, at in cells.items():
    par = at.cell.cellpar()
    at = at.copy(); at.set_cell(Cell.fromcellpar(par), scale_atoms=True)  # feos frame: a || x, b in xy
    n = [int(np.ceil(L / SPACING)) for L in par[:3]]
    t = time.time()
    v = lj_external_potential(grid_points(n).reshape(-1, 3) @ at.cell.array, at, 3.7039, 150.03).reshape(n)
    v = np.where(v < 50 * T, v, np.inf)
    n_prim = len(at) / 54
    grid, pot = periodic_grid(par, n), external_potential(v)
    print(f"--- {name}: cell {np.round(par, 2)}, grid {n}, min V {v.min():.0f} K ({time.time()-t:.0f}s)", flush=True)
    prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=1 * si.BAR), pot).solve()
    p, step, reported = 1.0, 0.3, set()
    while step > 1e-5 and p < 100:
        q = min(100.0, p * np.exp(step))
        try:
            prof = feos.PoreProfile(grid, feos.State(func, T * si.KELVIN, pressure=q * si.BAR), pot,
                                    density=prof.density).solve(feos.DFTSolver().newton(max_iter=50))
            p, step = q, step * 1.5
            for mark in [1, 3, 5, 8, 10, 20, 50, 100]:
                if p >= mark and mark not in reported:
                    reported.add(mark)
                    print(f"   p {p:7.2f} bar  N {float(prof.total_moles / si.MOL) * NAV / n_prim:6.3f} per prim. cell"
                          f"  max n3 {prof.weighted_densities[0][1].max():.3f}", flush=True)
        except Exception:  # noqa: BLE001
            step /= 2
    print(f"   adsorption branch ends at {p:.3f} bar ({'reached 100 bar' if p >= 100 else 'turning point'}), "
          f"{time.time()-t:.0f}s", flush=True)
