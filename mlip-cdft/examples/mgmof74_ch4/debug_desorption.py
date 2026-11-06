"""Does the desorption branch (from 100 bar down) converge? -> hysteresis / first-order transition test."""
import sys
import time

import numpy as np

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, interpolate_periodic  # noqa: E402

res = VextResult.load("vext_MH_smoke.npz")
T, mass = 298.0, 728.124
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
n = [int(np.ceil(L / 0.3)) for L in res.cellpar[:3]]
bv = interpolate_periodic(res.vext(T) / T, n, cap=50.0)
v = np.where(bv >= 50, np.inf, bv * T)
p = np.geomspace(0.1, 100, 25)
t = time.time()
iso = isotherm(func, v, res.cellpar, T, p, mass, branch="desorption")
ok = np.isfinite(iso["mol_per_kg"])
print(f"desorption: {ok.sum()}/{len(p)} converged ({time.time()-t:.0f}s)", flush=True)
for pb, nk, nm in zip(iso["pressure_bar"], iso["mol_per_kg"], iso["molecules_per_cell"]):
    print(f"   {pb:8.3f} bar  {nk:7.3f} mol/kg  {nm:6.3f}/cell", flush=True)
