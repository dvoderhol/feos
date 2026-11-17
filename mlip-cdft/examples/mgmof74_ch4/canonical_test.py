"""Trace N(p) through the branch end with fixed molecule numbers (CH4, standard UFF-LJ on the direct 0.3 A grid)."""
import sys
import numpy as np
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import _Pore3DBase, canonical_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult  # noqa: E402
T = 298.0
res = VextResult.load("vext_MH_smoke.npz")
v = np.load("v_lj_uff_0.3A.npy"); v = np.where(v < 50 * T, v, np.inf)
pore = _Pore3DBase(res.cellpar, v, 728.124)
func = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
r = canonical_isotherm(func, pore, T, np.arange(0.5, 14.01, 0.25))
for n, p in zip(r["molecules_per_cell"], r["pressure_bar"]):
    print(f"N {n:6.3f}/cell   p {p:10.4f} bar", flush=True)
print("stopped:", r["failed"])
dp = np.diff(r["pressure_bar"]); print("p(N) monotonic:", bool((dp > 0).all()), " max p:", r["pressure_bar"].max() if len(dp) else None)
