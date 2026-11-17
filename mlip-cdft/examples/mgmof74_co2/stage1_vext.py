"""Stage 1 (GPU): uMOF-MH insertion energies of CO2 in uMOF-MH-relaxed Mg-MOF-74 on a feos grid.

Usage: python stage1_vext.py <spacing_A> <n_orientations> <out.npz> [random|linear] [uMOF-MH|uMOF-POLAR]
The framework is the structure relaxed with the same model (relaxed_cifs/MgMOF74_prim_<model>.cif).
CO2 geometry is relaxed with the same model; the reference point is the C atom (molecular centre).
"""
import sys

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.vext import center_on_inversion, compute_vext, n_points_for_spacing, orientations  # noqa: E402
from mace_calc_helper import make_calc, prep_atoms_for_polar  # noqa: E402

spacing, n_orient, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else "random"
model_name = sys.argv[5] if len(sys.argv) > 5 else "uMOF-MH"
model = f"/local/home/dvoderhol/umof_lammps/models/{model_name}.model"

co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]], cell=[25] * 3, pbc=True)
co2.positions += 12.5
calc, is_polar = make_calc(model, "r2scan+d4", "float64", "cuda")
prep_atoms_for_polar(co2, is_polar)
co2.calc = calc
BFGS(co2, logfile=None).run(fmax=1e-4)
co2 = Atoms("CO2", positions=co2.positions - co2.positions[0])
print(f"CO2 relaxed with {model_name}: C-O = {np.linalg.norm(co2.positions[1]):.4f} A", flush=True)
del calc

uc, shift = center_on_inversion(read(f"/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_{model_name}.cif"))
eng = InsertionEngine(model, uc.repeat((2, 1, 1)), co2, dtype="float32", cueq=True)
n = n_points_for_spacing(uc, spacing)
res = compute_vext(eng, uc, n, orientations(kind, n_orient, seed=0), ref_index=0, batch_size=48)
res.info.update(model=model_name, guest="CO2", orientation_kind=kind, C_O=float(np.linalg.norm(co2.positions[1])), spacing=spacing,
                inversion_shift=None if shift is None else shift.tolist(), framework_mass_amu=float(uc.get_masses().sum()))
res.save(out)
v = res.vext(298.0)
print(res.info)
print(f"V_ext(298 K): min {v[np.isfinite(v)].min():.1f} K = {v[np.isfinite(v)].min()*8.314462e-3:.1f} kJ/mol, "
      f"accessible fraction {np.isfinite(v).mean():.3f}")
