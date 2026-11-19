"""Validation of delta = -1/2 f^T H+ f against explicit framework relaxation at fixed guest position and orientation
(notes, Remark 'Temperature and relaxation energy'): E_relax(x) - E0(x) = delta(x) + anharmonic/non-linear terms.
float64 on CPU (ASE + MACE calculator), framework fully relaxed (BFGS, fmax 1e-3 eV/A), CO2 fixed, unweighted framework centroid fixed
(the translation-free subspace U on which delta is defined; a free framework translation would move the guest).
Also prints the (A5) checks of the 2x1x1 Hessian. Writes validate_delta_relax.json."""
import json
import sys
import time

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from mace_calc_helper import make_calc  # noqa: E402
from feos_mlip.flex import HarmonicFramework, orientation_set, refined_rows  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion, grid_points  # noqa: E402


class FixGuestAndFrameworkCentroid:
    """Guest atoms fixed; framework relaxes in the translation-free subspace U of the notes (unweighted centroid
    of the framework fixed): forces on framework atoms minus their mean, guest forces zero."""

    def __init__(self, n_fw):
        self.n_fw = n_fw

    def adjust_positions(self, atoms, new):
        n = self.n_fw
        new[n:] = atoms.positions[n:]
        new[:n] -= (new[:n] - atoms.positions[:n]).mean(0)

    def adjust_forces(self, atoms, forces):
        n = self.n_fw
        forces[:n] -= forces[:n].mean(0)
        forces[n:] = 0.0

    def get_removed_dof(self, atoms):
        return 3 * (len(atoms) - self.n_fw) + 3

    def todict(self):
        return {"name": "FixGuestAndFrameworkCentroid", "kwargs": {"n_fw": self.n_fw}}

hz = np.load("hessian_2x1x1.npz")
fw = HarmonicFramework(hz["H"])
print("Hessian checks:", fw.summary(), flush=True)
res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat((2, 1, 1)); host.wrap()
assert np.allclose(host.positions, hz["positions"])
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, res.info["C_O"]], [0, 0, -res.info["C_O"]]])
calc, _ = make_calc("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", "r2scan+d4", "float64", "cpu")
h = host.copy(); h.calc = calc; E_h = h.get_potential_energy(); F_h = h.get_forces()
g = co2.copy(); g.positions -= g.positions.mean(0); g.set_cell([30] * 3); g.pbc = True; g.positions += 15
g.calc = calc; E_g = g.get_potential_energy()
U = np.nan_to_num(res.U, nan=np.inf)
ref = refined_rows(res)
order = np.argsort(U.ravel())
picks, seen = [], set()
for flat in order:                                    # 6 deepest distinct points, best orientation each
    i, q = np.unravel_index(flat, U.shape)
    if i in seen:
        continue
    seen.add(i); picks.append((i, q))
    if len(picks) == 6:
        break
rng = np.random.default_rng(3)
for lo, hi in [(-0.35, -0.2), (-0.2, -0.05), (-0.05, 0.1)]:   # moderate, weak, slightly repulsive
    cand = np.argwhere((U > lo) & (U < hi))
    picks += [tuple(c) for c in cand[rng.choice(len(cand), 2, replace=False)]]
gpos0 = co2.positions - co2.positions[0]
out = []
for i, q in picks:
    R = orientation_set(res, refined=bool(ref[i]))[q]
    G = gpos0 @ R.T + grid_points(res.n).reshape(-1, 3)[res.irr[i]] @ uc.cell.array
    a = host + Atoms("CO2", positions=G); a.set_cell(host.cell); a.pbc = True; a.calc = calc
    a.set_constraint(FixGuestAndFrameworkCentroid(len(host)))
    E0 = a.get_potential_energy() - E_h - E_g
    f = (a.get_forces(apply_constraint=False)[:len(host)] - F_h)
    delta = float(fw.delta(f[None])[0])
    t = time.time()
    BFGS(a, logfile=None).run(fmax=1e-3, steps=2000)
    E_rel = a.get_potential_energy() - E_h - E_g
    u = a.positions[:len(host)] - host.positions
    u_harm = fw.displacement(f[None])[0]
    u_rel = u - u.mean(0)                                # compare in the translation-free subspace
    rec = dict(row=int(i), col=int(q), E0_stored=float(res.U[i, q]), E0=E0, delta=delta, relax=E_rel - E0,
               ratio=(E_rel - E0) / delta, max_u_relax=float(np.linalg.norm(u_rel, axis=1).max()),
               max_u_harm=float(np.linalg.norm(u_harm, axis=1).max()),
               corr_u=float(np.dot(u_rel.ravel(), u_harm.ravel()) / np.linalg.norm(u_rel) / np.linalg.norm(u_harm)),
               relax_s=round(time.time() - t, 1))
    out.append(rec)
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
json.dump(dict(hessian=fw.summary(), configs=out), open("validate_delta_relax.json", "w"), indent=1)
