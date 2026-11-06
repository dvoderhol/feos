"""Stage 1 (GPU): MLIP V_ext for CH4 in Mg-MOF-74 (uMOF-relaxed primitive cell) on a feos grid.

Usage: python stage1_vext.py <uMOF-MH|uMOF-POLAR> <spacing_A> <n_orientations> <out.npz>
"""
import sys

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.vext import center_on_inversion, compute_vext, n_points_for_spacing, orientations  # noqa: E402

model, spacing, n_orient, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
uc, shift = center_on_inversion(read(f"/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_{model}.cif"))
host = uc.repeat((2, 1, 1))  # guest images >= 13.7 A apart (receptive field 12 A)
# CH4 geometry relaxed with uMOF-MH (C-H 1.0887 A); reference point = C atom (PC-SAFT segment centre, m = 1)
d = 1.0887 / np.sqrt(3)
ch4 = Atoms("CH4", positions=[[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
eng = InsertionEngine(f"/local/home/dvoderhol/umof_lammps/models/{model}.model", host, ch4,
                      dtype="float32", cueq=True)
n = n_points_for_spacing(uc, spacing)
res = compute_vext(eng, uc, n, orientations("random", n_orient, seed=0), ref_index=0, batch_size=48)
res.info.update(model=model, spacing=spacing, inversion_shift=None if shift is None else shift.tolist(),
                framework_mass_amu=float(uc.get_masses().sum()))
res.save(out)
v = res.vext(298.0)
print(res.info)
print(f"V_ext(298 K): min {np.nanmin(v[np.isfinite(v)]):.1f} K, accessible fraction {np.isfinite(v).mean():.3f}")
