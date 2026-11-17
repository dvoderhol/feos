"""Orientation convergence: re-evaluate the strongly attractive grid points (best U < -0.30 eV) of the 48-direction
CO2 V_ext with 96 and 192 linear-molecule directions; report K_H and save the refined results."""
import sys
import numpy as np
from ase import Atoms
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import henry_from_vext  # noqa: E402
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion, orientations, refine_orientations  # noqa: E402

T = 298.0
res = VextResult.load("vext_CO2_MH_0.4A_lin48.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, res.info["C_O"]], [0, 0, -res.info["C_O"]]])
eng = InsertionEngine("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", uc.repeat((2, 1, 1)), co2,
                      dtype="float32", cueq=True)
mass = res.info["framework_mass_amu"]
print(f"48 directions: K_H {henry_from_vext(res.vext(T), res.cellpar, T, mass):.4e}; "
      f"points with best U < -0.30 eV: {(np.min(res.U, axis=1) < -0.30).sum()} of {len(res.U)}", flush=True)
for n in (96, 192, 384):
    ref = refine_orientations(res, eng, uc, orientations("linear", n), u_threshold=-0.30)
    v = ref.vext(T)
    print(f"{n} directions on the deep points: K_H {henry_from_vext(v, res.cellpar, T, mass):.4e}; "
          f"min V {v[np.isfinite(v)].min():.1f} K", flush=True)
    ref.save(f"vext_CO2_MH_0.4A_lin48_refined{n}.npz")
