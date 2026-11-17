"""Production CO2 V_ext (uMOF-POLAR, 298 K): 48 linear-molecule directions on all grid points, 192 directions on the
strongly attractive points (best U < -0.30 eV; converged: 96 vs 192 differ by 0.1 % in K_H), guest-atom overlap
guard 1.8 A everywhere (1.4 A admits uMOF-MH extrapolation holes). Writes vext_CO2_POLAR_production.npz."""
import sys
from ase import Atoms
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import henry_from_vext  # noqa: E402
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.vext import VextResult, apply_overlap_guard, center_on_inversion, orientations, refine_orientations  # noqa: E402

base = VextResult.load("vext_CO2_POLAR_0.4A_lin48.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-POLAR.cif"))
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, base.info["C_O"]], [0, 0, -base.info["C_O"]]])
host = uc.repeat((2, 1, 1))
guarded = apply_overlap_guard(base, uc, co2, host, 1.8)
eng = InsertionEngine("/local/home/dvoderhol/umof_lammps/models/uMOF-POLAR.model", host, co2, dtype="float32", cueq=True)
prod = refine_orientations(guarded, eng, uc, orientations("linear", 192), u_threshold=-0.30, r_atom_min=1.8)
prod.save("vext_CO2_POLAR_production.npz")
print(f"production V_ext: K_H {henry_from_vext(prod.vext(298.0), prod.cellpar, 298.0, base.info['framework_mass_amu']):.4e} "
      f"mol/(kg Pa), refined points {prod.info['refined_points']}")
