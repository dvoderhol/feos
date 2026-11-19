"""Timing benchmark + MLIP grid-spacing convergence of the CO2 V_ext (uMOF-MH), same protocol as the production
(make_production_vext.py): 48 linear-molecule orientations on all points, overlap guard 1.8 A, 192 orientations on
points with best U < -0.30 eV. Run ALONE on an otherwise idle GPU. Usage: python bench_vext.py <spacing_A>.
Writes vext_CO2_MH_<spacing>A_bench.npz and bench_vext_<spacing>.json (wall times per stage, K_H)."""
import json
import socket
import subprocess
import sys
import time

import torch
from ase import Atoms
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import henry_from_vext  # noqa: E402
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.vext import (center_on_inversion, compute_vext, n_points_for_spacing, orientations,  # noqa: E402
                            refine_orientations)

h = float(sys.argv[1])
T = 298.0
t0 = time.time()
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, 1.1677672052747283], [0, 0, -1.1677672052747283]])
eng = InsertionEngine("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", uc.repeat((2, 1, 1)), co2,
                      dtype="float32", cueq=True)
t_setup = time.time() - t0
n = n_points_for_spacing(uc, h)
t1 = time.time()
base = compute_vext(eng, uc, n, orientations("linear", 48), r_ref_min=2.0, r_atom_min=1.8, progress=False)
t_base = time.time() - t1
t2 = time.time()
prod = refine_orientations(base, eng, uc, orientations("linear", 192), u_threshold=-0.30, r_atom_min=1.8)
t_ref = time.time() - t2
prod.save(f"vext_CO2_MH_{h}A_bench.npz")
mass = 728.124
out = dict(spacing=h, n_grid=list(map(int, n)), K_H=henry_from_vext(prod.vext(T), prod.cellpar, T, mass),
           K_H_base48=henry_from_vext(base.vext(T), base.cellpar, T, mass),
           insertions_base=base.info["n_insertions"], refined_points=prod.info["refined_points"],
           wall_setup_s=round(t_setup, 1), wall_base_s=round(t_base, 1), wall_refine_s=round(t_ref, 1),
           wall_total_s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
           gpu_processes=subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
                                         f"--id={torch.cuda.current_device()}"], capture_output=True, text=True).stdout)
json.dump(out, open(f"bench_vext_{h}.json", "w"), indent=1)
print(out)
