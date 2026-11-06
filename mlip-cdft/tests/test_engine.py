"""Check InsertionEngine (batched, energy-only, GPU edges) against the plain ASE MACE calculator."""
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
from feos_mlip.engine import InsertionEngine  # noqa: E402
from mace_calc_helper import make_calc, prep_atoms_for_polar  # noqa: E402

MODELS = "/local/home/dvoderhol/umof_lammps/models"
CIFS = "/local/home/dvoderhol/umof_lammps/relaxed_cifs"
d = 1.0887 / np.sqrt(3)
CH4 = Atoms("CH4", positions=[[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
rng = np.random.default_rng(3)


def random_configs(host, n, rmin=2.2):
    out, H = [], host.cell.array
    while len(out) < n:
        q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, e = q
        R = np.array([[a*a+b*b-c*c-e*e, 2*(b*c-a*e), 2*(b*e+a*c)], [2*(b*c+a*e), a*a-b*b+c*c-e*e, 2*(c*e-a*b)],
                      [2*(b*e-a*c), 2*(c*e+a*b), a*a-b*b-c*c+e*e]])
        frac = rng.random(3)
        if len(out) < 3:  # force molecules straddling the cell boundary
            frac[len(out)] = 0.999
        g = CH4.positions @ R.T + frac @ H
        dd = g[:, None] - host.positions[None]
        f = dd @ np.linalg.inv(H); f -= np.round(f)
        if np.sqrt(((f @ H) ** 2).sum(-1)).min() > rmin:
            out.append(g)
    return np.array(out)


for name in ["uMOF-MH", "uMOF-POLAR"]:
    host = read(f"{CIFS}/MgMOF74_prim_{name}.cif").repeat((2, 1, 1))
    cfgs = random_configs(host, 32)
    # reference: plain ASE calculator, float64, e3nn, full periodic system
    calc, pol = make_calc(f"{MODELS}/{name}.model", "r2scan+d4", "float64", "cuda")
    h = host.copy(); prep_atoms_for_polar(h, pol); h.calc = calc; e_h = h.get_potential_energy()
    iso = CH4.copy(); iso.set_cell([30] * 3); iso.pbc = True; iso.positions += 15
    prep_atoms_for_polar(iso, pol); iso.calc = calc; e_g = iso.get_potential_energy()
    ref = []
    for g in cfgs[:8]:
        c = host + Atoms("CH4", positions=g); c.set_cell(host.cell); c.pbc = True
        prep_atoms_for_polar(c, pol); c.calc = calc
        ref.append(c.get_potential_energy() - e_h - e_g)
    ref = np.array(ref)
    del calc
    for dtype, cueq in [("float64", False), ("float32", False), ("float32", True)]:
        eng = InsertionEngine(f"{MODELS}/{name}.model", host, CH4, dtype=dtype, cueq=cueq)
        u = eng.energies(cfgs[:8])
        bs = None
        bs = 32 if cueq else 8
        t = time.time(); n_rep = 4
        for _ in range(n_rep):
            for k in range(0, len(cfgs), bs):
                eng.energies(cfgs[k:k + bs])
        rate = n_rep * len(cfgs) / (time.time() - t)
        print(f"{name:10s} {dtype} cueq={cueq!s:5s} node_energies={eng.use_node_energies!s:5s} "
              f"max|U-U_ref|={np.abs(u-ref).max()*1000:8.4f} meV  (batch {bs}: {rate:6.0f} configs/s)  "
              f"U_ref[:4]={np.round(ref[:4], 4)}", flush=True)
        tol_mev = 1e-3 if dtype == "float64" else (0.1 if eng.use_node_energies else 1.0)
        assert np.abs(u - ref).max() * 1000 < tol_mev, f"{name} {dtype} cueq={cueq}: insertion energies off"
        del eng
        torch.cuda.empty_cache()
