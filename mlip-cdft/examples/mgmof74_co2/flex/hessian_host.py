"""Hessian (float64, autograd) of the uMOF-MH host used for V_ext: Mg-MOF-74 primitive cell (uMOF-MH relaxed),
shifted onto its inversion centre, repeated 2x1x1 (or as given). Writes hessian_<rep>.npz with H (3M x 3M, eV/A^2),
positions, cell, numbers, residual forces and the wall time.  Usage: python hessian_host.py [na nb nc] [device]"""
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from mace_calc_helper import make_calc  # noqa: E402
from feos_mlip.vext import center_on_inversion  # noqa: E402

rep = tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 else (2, 1, 1)
device = sys.argv[4] if len(sys.argv) >= 5 else "cpu"
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat(rep)
host.wrap()                       # same as InsertionEngine (indices unchanged)
calc, _ = make_calc("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", "r2scan+d4", "float64", device)
host.calc = calc
f0 = host.get_forces()
t = time.time()
H = np.asarray(calc.get_hessian(atoms=host)).reshape(3 * len(host), 3 * len(host))
wall = time.time() - t
tag = "x".join(map(str, rep))
np.savez(f"hessian_{tag}.npz", H=H, positions=host.positions, cell=host.cell.array, numbers=host.numbers,
         forces=f0, wall_s=wall, device=device)
print(f"{tag}: N={len(host)} Hessian {H.shape} in {wall:.1f} s on {device}; max|F| {np.abs(f0).max():.2e} eV/A; "
      f"asym {np.abs(H - H.T).max():.2e}")
