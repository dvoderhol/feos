"""Core checks of feos_mlip.flex on the production data (CPU only):
1. delta statistics and K_H (rigid vs flexible) for E_c = 0, 10, 20 kT; rigid K_H with E_c must equal the production
   value (configs above E_c carry negligible weight).
2. basins (steepest descent on V with delta).
3. functional derivative: v_w(p) = (1/(2 rho dV)) dF_w/dpi_p by central finite differences (2 primitive copies in the
   supercell), for random pairs, both representative and inversion-image points, at a random density and ubar from it.
Writes test_flex_core.json."""
import json
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import henry_from_vext  # noqa: E402
from feos_mlip.flex import FlexCoupling, FlexibleVext, HarmonicFramework  # noqa: E402
from feos_mlip.vext import KB_EV, VextResult, center_on_inversion  # noqa: E402

T = 298.0
kT = KB_EV * T
res = VextResult.load("../vext_CO2_MH_production.npz")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
host = uc.repeat((2, 1, 1)); host.wrap()
fw = HarmonicFramework(np.load("hessian_2x1x1.npz")["H"])
t = time.time(); cp = FlexCoupling.load("coupling_CO2_MH_20kT.npz"); t_load = time.time() - t
mass = res.info["framework_mass_amu"]
out = dict(K_H_production=henry_from_vext(res.vext(T), res.cellpar, T, mass))
print(f"K_H production (rigid, all configs) {out['K_H_production']:.5e}", flush=True)
for n in (20, 10, 0):
    t = time.time()
    fv = FlexibleVext(res, cp, fw, uc, host, T, e_cut=n * kT)
    vr, vf = fv.v_rigid(), fv.v_flex()
    kr, kf = henry_from_vext(vr, res.cellpar, T, mass), henry_from_vext(vf, res.cellpar, T, mass)
    d = fv.delta
    rec = dict(e_cut_kT=n, n_cfg=int(len(d)), K_H_rigid=kr, K_H_flex=kf, ratio=kf / kr,
               delta_min=float(d.min()), delta_p1=float(np.percentile(d, 1)), delta_median=float(np.median(d)),
               delta_min_E0_below_m03=float(d[fv.E0 < -0.3].min()), delta_mean_E0_below_m03=float(d[fv.E0 < -0.3].mean()),
               build_s=round(time.time() - t, 1))
    out[f"E_c_{n}kT"] = rec
    print(rec, flush=True)
fv = FlexibleVext(res, cp, fw, uc, host, T, e_cut=10 * kT)
print(f"permutation mismatch inversion {fv.err_inv:.2e} A, translations {len(fv.translations)}", flush=True)
label, minima = fv.basins()
vflex = fv.v_flex().reshape(-1)
depth = np.sort(vflex[minima] * KB_EV)
out["basins"] = dict(n=int(len(minima)), deepest_eV=[float(x) for x in depth[:10]],
                     n_below_m02eV=int((depth < -0.2).sum()), n_above_0=int((depth > 0).sum()))
print("basins:", out["basins"], flush=True)
# 3. functional derivative test
rng = np.random.default_rng(0)
rho = np.where(np.isfinite(vflex), rng.random(fv.n_full) * 0.02, 0.0)          # molecules / A^3
e1 = (fv.E0 + fv.delta)[fv.pair_cfg]
_, pi = fv._average(e1)
Fk = fv.basin_forces(rho, pi, label)
ubar = fw.displacement(fv.total_force(Fk)[None] - Fk)
vw = fv.v_w(ubar, label)
checks = []
for inv in (False, True):
    cand = np.nonzero((fv.pair_inv == inv) & (pi > 1e-3))[0]
    for p in rng.choice(cand, 4, replace=False):
        h = 1e-3 * pi[p]
        Fp = []
        for s in (+1, -1):
            pp = pi.copy(); pp[p] += s * h
            Fp.append(fv.free_energy_w(rho, label, pp))
        num = (Fp[0] - Fp[1]) / (2 * h) / (2 * rho[fv.pair_point[p]] * fv.dV)
        checks.append(dict(inverted=inv, v_w=float(vw[p]), numerical=float(num), rel_err=float(abs(num - vw[p]) / abs(vw[p]))))
        print(checks[-1], flush=True)
out["derivative_checks"] = checks
out["coupling_load_s"] = round(t_load, 1)
with open("test_flex_core.json", "w") as f:
    json.dump(out, f, indent=1)
