"""Independent K_H check for the published NequIP V_ext: random-insertion Widom (uniform position in the unit cell,
uniform random orientation) with the same engine and the paper's overlap rule; no grid, no orientation sets.
usage: python widom_check.py <model key> <n_insertions>"""
import json, sys, time
import numpy as np
from ase import Atoms
from ase.io import read
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.engine_nequip import NequIPInsertionEngine
from feos_mlip.vext import center_on_inversion, min_distance_to_host, pair_thresholds_vdw
import make_vext_nequip_cfg as cfg
key, N = sys.argv[1], int(sys.argv[2])
model, struct, gt, vdw, rep, use_sym, C_O = cfg.MODELS[key]
uc = center_on_inversion(read(struct))[0] if use_sym else read(struct)
host = uc.repeat(rep); mass = float(uc.get_masses().sum())
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, C_O], [0, 0, -C_O]])
eng = NequIPInsertionEngine(str(model), host, co2, guest_types=gt, device="cuda")
thr = pair_thresholds_vdw(["C", "O", "O"], uc, vdw["delta"], vdw["overrides"])
sym = np.array(host.get_chemical_symbols())
kT = 8.617333262e-5 * 298.0; rng = np.random.default_rng(11)
w = []; t = time.time(); B = 256
for s in range(0, N, B):
    nb = min(B, N - s)
    c = rng.random((nb, 3)) @ uc.cell.array
    R = Rotation.random(nb, random_state=rng.integers(1e9)).as_matrix()
    g = np.einsum("bij,aj->bai", R, co2.positions) + c[:, None, :]
    bad = np.zeros(nb, bool)
    for e, th in thr.items():
        d = min_distance_to_host(g.reshape(-1, 3), host[sym == e]).reshape(nb, 3)
        bad |= np.any(d < th[None], axis=1)
    u = np.full(nb, np.inf)
    if (~bad).any():
        u[~bad] = eng.energies(g[~bad])
    u[u < -1.5] = np.inf      # same extrapolation guard as the grid V_ext
    w.append(np.exp(-u / kT))
w = np.concatenate(w)
blocks = np.array([b.mean() for b in np.array_split(w, 20)])
fac = uc.get_volume() * 1e-30 / (mass * 1.66053906660e-27 * 8.31446262 * 298.0)
kh, err = w.mean() * fac, blocks.std(ddof=1) / np.sqrt(20) * fac
print(f"{key}: Widom {N} insertions ({time.time()-t:.0f} s): K_H = {kh:.4e} +- {err:.2e} mol/(kg Pa) (paper overlap rule)", flush=True)
json.dump({"key": key, "N": N, "K_H": kh, "K_H_err": err}, open(f"widom_{key}.json", "w"), indent=1)
