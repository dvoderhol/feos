"""Continue the fixed-N (PoreSpecification.Moles) trace of m = 1 CO2 through the van der Waals loop with small steps
(dN = 0.02 per cell; on failure: halve the step, then try a damped log-Picard + Newton chain). Prints N, p(N) and the
grand potential, so that the equilibrium (equal grand potential) transition can be located."""
import json
import sys
import time
import numpy as np
import si_units as si
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T, NAV = 298.0, 6.02214076e23
which = sys.argv[1]
par = json.load(open("/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"))[0]
func = pcsaft(["carbon dioxide"], "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
if which == "uff":
    pore, p0 = LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T), np.geomspace(1e-3, 0.5, 12)
else:
    res = VextResult.load("vext_CO2_MH_0.4A_lin48_guard1.8.npz")
    pore, p0 = MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0), np.geomspace(1e-5, 1.2e-3, 12)
iso = adaptive_isotherm(func, pore, T, p0)
prof = iso["profiles"][-1]
n = float(prof.total_moles / si.MOL) * NAV
solvers = [feos.DFTSolver().newton(max_iter=200),
           feos.DFTSolver().picard_iteration(log=True, max_iter=300, tol=1e-7, damping_coefficient=0.05).newton(max_iter=200)]
step, t0, rows = 0.02, time.time(), []
while n < 40 and step > 1e-4 and time.time() - t0 < 6 * 3600:
    spec = feos.PoreSpecification.Moles(np.array([(n + step) / NAV]) * si.MOL)
    new = None
    for solver in solvers:
        try:
            new = pore.initialize(prof.bulk, density=prof.density, specification=spec).solve(solver)
            break
        except Exception:  # noqa: BLE001
            continue
    if new is None:
        step /= 2
        continue
    prof, n = new, n + step
    step = min(step * 1.5, 0.2)
    p = float(prof.bulk.pressure() / si.BAR)
    omega = float(prof.grand_potential / si.JOULE) if prof.grand_potential is not None else float("nan")
    rows.append((n, p, omega))
    print(f"N {n:7.3f}/cell  p {p:12.6g} bar  mmol/g {pore.per_mass(prof.total_moles):7.3f}  Omega {omega:.6e} J  "
          f"max n3 {prof.weighted_densities[0][1].max():.3f}", flush=True)
print(f"stopped at N = {n:.3f} (step {step:.2e})", flush=True)
json.dump(rows, open(f"canonical_loop_{which}.json", "w"))
