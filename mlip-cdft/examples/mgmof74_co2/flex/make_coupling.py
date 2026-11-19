"""Coupling forces f(x) = F_host(host + CO2) - F_host(host) (uMOF-MH, float32/cuEquivariance, same engine, host and
orientation sets as vext_CO2_MH_production.npz) for every stored configuration with E0 < 20 kT (298 K).
Writes coupling_CO2_MH_20kT.npz (feos_mlip.flex.FlexCoupling)."""
import sys
import time

from ase import Atoms
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.engine import InsertionEngine  # noqa: E402
from feos_mlip.flex import compute_coupling  # noqa: E402
from feos_mlip.vext import KB_EV, VextResult, center_on_inversion  # noqa: E402

res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
co2 = Atoms("CO2", positions=[[0, 0, 0], [0, 0, res.info["C_O"]], [0, 0, -res.info["C_O"]]])
t0 = time.time()
eng = InsertionEngine("/local/home/dvoderhol/umof_lammps/models/uMOF-MH.model", uc.repeat((2, 1, 1)), co2,
                      dtype="float32", cueq=True)
print(f"engine setup {time.time() - t0:.1f} s", flush=True)
cp = compute_coupling(res, eng, uc, e_cut=20 * KB_EV * 298.0, batch_size=int(sys.argv[1]) if len(sys.argv) > 1 else 64)
cp.info["engine_setup_s"] = round(time.time() - t0 - cp.info["wall_s"], 1)
cp.save("coupling_CO2_MH_20kT.npz")
print(cp.info, flush=True)
