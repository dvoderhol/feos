"""Flexible-framework Widom reference for delta (notes, Practical evaluation): classical Langevin MD of the empty
uMOF-MH framework (2x1x1, fixed cell, 298 K, dt 0.5 fs, 5 ps equilibration + 40 ps, snapshots every 20 fs),
snapshots re-centred onto the translation-free subspace U (unweighted framework centroid = that of R0). At fixed
guest configurations x:   delta_MD(x) = -kT ln < exp(-beta [E(x, u) - E0(x)]) >_u   (exact incl. anharmonicity,
classical)  vs  delta(x) = -1/2 f^T H+ f  (harmonic, linear coupling). Writes flex_widom.npz/.json."""
import json
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from mace_calc_helper import make_calc  # noqa: E402
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.flex import HarmonicFramework, orientation_set, refined_rows  # noqa: E402
from feos_mlip.vext import KB_EV, VextResult, center_on_inversion, grid_points  # noqa: E402

T = 298.0
kT = KB_EV * T
res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat((2, 1, 1)); host.wrap()
R0 = host.positions.copy()
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, res.info["C_O"]], [0, 0, -res.info["C_O"]]])
fw = HarmonicFramework(np.load("hessian_2x1x1.npz")["H"])

# --- configurations: 6 deepest distinct points (best orientation) + 34 random accessible configs with E0 < 0
U = np.nan_to_num(res.U, nan=np.inf)
ref = refined_rows(res)
picks, seen = [], set()
for flat in np.argsort(U.ravel()):
    i, q = np.unravel_index(flat, U.shape)
    if i not in seen:
        seen.add(i); picks.append((int(i), int(q)))
    if len(picks) == 6:
        break
cand = np.argwhere(U < 0.0)
rng = np.random.default_rng(11)
picks += [tuple(int(v) for v in c) for c in cand[rng.choice(len(cand), 34, replace=False)]]
gpos0 = co2.positions - co2.positions[0]
G = np.array([gpos0 @ orientation_set(res, refined=bool(ref[i]))[q].T + grid_points(res.n).reshape(-1, 3)[res.irr[i]]
              @ uc.cell.array for i, q in picks])

eng = InsertionEngine("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", host, co2, dtype="float32", cueq=True)
E0, f = eng.energies_forces(G)
delta = fw.delta(f)

# --- MD of the empty framework
calc, _ = make_calc("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", "r2scan+d4", "float32", "cuda")
md_atoms = host.copy(); md_atoms.calc = calc
MaxwellBoltzmannDistribution(md_atoms, temperature_K=T, rng=np.random.default_rng(5))
dyn = Langevin(md_atoms, 0.5 * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(6))
t0 = time.time()
dyn.run(10000)                                   # 5 ps equilibration
t_eq = time.time() - t0
snaps, temps = [], []
for k in range(2000):                            # 40 ps, every 40 steps = 20 fs
    dyn.run(40)
    u = md_atoms.positions - R0
    snaps.append(R0 + u - u.mean(0))
    temps.append(md_atoms.get_temperature())
t_md = time.time() - t0
snaps = np.array(snaps)
print(f"MD: {t_md:.0f} s ({t_eq:.0f} s equilibration), <T> = {np.mean(temps):.1f} K", flush=True)

# --- insertion energies on every snapshot
dE = np.empty((len(snaps), len(G)))
t1 = time.time()
for k, R in enumerate(snaps):
    eng.set_host_positions(R)
    dE[k] = eng.energies(G) - E0
t_ins = time.time() - t1
x = np.exp(-dE / kT)
blocks = np.array([b.mean(0) for b in np.array_split(x, 10)])
mean, err = x.mean(0), blocks.std(0, ddof=1) / np.sqrt(10)
d_md = -kT * np.log(mean)
d_err = kT * err / mean
np.savez("flex_widom.npz", snaps=snaps.astype(np.float32), dE=dE, E0=E0, delta=delta, picks=np.array(picks))
out = dict(T=T, md_s=round(t_md, 1), insert_s=round(t_ins, 1), mean_T=float(np.mean(temps)), configs=[
    dict(row=i, col=q, E0=float(e), delta=float(d), delta_md=float(m), delta_md_err=float(s),
         mean_dE=float(dE[:, k].mean()), std_dE=float(dE[:, k].std()))
    for k, ((i, q), e, d, m, s) in enumerate(zip(picks, E0, delta, d_md, d_err))])
json.dump(out, open("flex_widom.json", "w"), indent=1)
for c in out["configs"]:
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()))
print(f"insertions: {t_ins:.0f} s", flush=True)
