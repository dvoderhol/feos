"""Baseline (standard approach): UFF LJ 12-6 solid-fluid V_ext for PC-SAFT CH4 on the same feos grid,
plus a comparison of the MLIP V_ext with it. Then adsorption isotherms 0.1-100 bar for both.

UFF (Rappe 1992; values typed from memory - verify): x_i [A], D_i [kcal/mol]; sigma = x 2^(-1/6), eps/k = D 503.22 K.
Lorentz-Berthelot with PC-SAFT methane (gross2001: sigma 3.7039 A, eps/k 150.03 K, m = 1); cutoff 14 A, no tail.
"""
import sys
import time

import numpy as np
from ase.cell import Cell
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion, grid_points, interpolate_periodic  # noqa: E402

UFF = {"H": (2.886, 0.044), "C": (3.851, 0.105), "O": (3.500, 0.060), "Mg": (3.021, 0.111)}
SIG_F, EPS_F, RC = 3.7039, 150.03, 14.0
T, mass = 298.0, 728.124
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
res = VextResult.load("vext_MH_smoke.npz")
assert np.allclose(uc.cell.cellpar(), res.cellpar)
n = [int(np.ceil(L / 0.3)) for L in res.cellpar[:3]]

# standard LJ potential on the fine grid: explicit sum over lattice images within the cutoff
cell = Cell.fromcellpar(res.cellpar)  # same orientation convention as feos (a || x, b in xy)
uc.set_cell(cell, scale_atoms=True)
pts = grid_points(n).reshape(-1, 3) @ cell.array
sig_s = np.array([UFF[s][0] * 2 ** (-1 / 6) for s in uc.get_chemical_symbols()])
eps_s = np.array([UFF[s][1] * 503.22 for s in uc.get_chemical_symbols()])
sig, eps = 0.5 * (sig_s + SIG_F), np.sqrt(eps_s * EPS_F)
widths = cell.volume / np.linalg.norm(np.cross(cell.array[[1, 2, 0]], cell.array[[2, 0, 1]]), axis=1)
nr = np.ceil(RC / widths).astype(int) + 1
shifts = np.array([(i, j, k) for i in range(-nr[0], nr[0] + 1) for j in range(-nr[1], nr[1] + 1)
                   for k in range(-nr[2], nr[2] + 1)]) @ cell.array
atoms_img = (uc.positions[None] + shifts[:, None]).reshape(-1, 3)
sig_img, eps_img = np.tile(sig, len(shifts)), np.tile(eps, len(shifts))
v_lj = np.empty(len(pts))
t = time.time()
for s in range(0, len(pts), 512):
    r2 = ((pts[s:s + 512, None] - atoms_img[None]) ** 2).sum(-1)
    sr6 = (sig_img ** 2 / r2) ** 3
    v_lj[s:s + 512] = np.where(r2 < RC ** 2, 4 * eps_img * (sr6 ** 2 - sr6), 0.0).sum(1)
v_lj = v_lj.reshape(n)
print(f"LJ V_ext on {n} in {time.time()-t:.0f}s: min {v_lj.min():.1f} K", flush=True)

bv = interpolate_periodic(res.vext(T) / T, n, cap=50.0)
v_ml = np.where(bv >= 50, np.inf, bv * T)
both = np.isfinite(v_ml) & (v_lj < 50 * T)
print(f"MLIP V_ext: min {v_ml[np.isfinite(v_ml)].min():.1f} K; accessible (V<50kT) fraction MLIP {np.isfinite(v_ml).mean():.3f}, LJ {(v_lj < 50*T).mean():.3f}")
print(f"where both accessible: corr {np.corrcoef(v_ml[both], v_lj[both])[0,1]:.3f}, mean(MLIP-LJ) {np.mean(v_ml[both]-v_lj[both]):.1f} K")
for thr in [-3, -5, -7]:
    print(f"  volume fraction with beta*V < {thr}: MLIP {(v_ml/T < thr).mean():.4f}  LJ {(v_lj/T < thr).mean():.4f}")

func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
p = np.geomspace(0.1, 100, 16)
for name, v in [("standard UFF-LJ", np.where(v_lj < 50 * T, v_lj, np.inf)), ("MLIP uMOF-MH", v_ml)]:
    t = time.time()
    iso = isotherm(func, v, res.cellpar, T, p, mass)
    ok = np.isfinite(iso["mol_per_kg"])
    print(f"{name}: {ok.sum()}/{len(p)} converged ({time.time()-t:.0f}s)", flush=True)
    for pb, nk in zip(p, iso["mol_per_kg"]):
        print(f"   {pb:8.3f} bar  {nk:7.3f} mol/kg", flush=True)
np.save("v_lj_uff_0.3A.npy", v_lj)
