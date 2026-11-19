"""Infinite-dilution flexibility (delta) of CO2 in Mg-MOF-74 (uMOF-MH, 298 K): K_H rigid vs flexible as a function of
the harmonic validity cut u_cut (0.1/0.2/0.3 A) and the energy cut E_c (0/10/20 kT). For the configurations treated
rigidly (max|u*| > u_cut) a rough estimate of what is missed: delta_est = r(u*) delta with r interpolated from the
explicit relaxations (validate_delta_relax*.json). Writes flex_henry.json."""
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
cp = FlexCoupling.load("coupling_CO2_MH_20kT.npz")
mass = res.info["framework_mass_amu"]
rel = json.load(open("validate_delta_relax.json"))["configs"] + json.load(open("validate_delta_relax_large.json"))["configs"]
uh = np.array([c["max_u_harm"] for c in rel]); ratio = np.array([c["ratio"] for c in rel])
o = np.argsort(uh); uh, ratio = uh[o], ratio[o]
out = dict(K_H_production=henry_from_vext(res.vext(T), res.cellpar, T, mass), runs=[])
for e_n in (0, 10, 20):
    for u_cut in (None, 0.1, 0.2, 0.3):
        t = time.time()
        fv = FlexibleVext(res, cp, fw, uc, host, T, e_cut=e_n * kT, u_cut=u_cut)
        k_r = henry_from_vext(fv.v_rigid(), res.cellpar, T, mass)
        k_f = henry_from_vext(fv.v_flex(), res.cellpar, T, mass)
        rec = dict(e_cut_kT=e_n, u_cut=u_cut, K_H_rigid=k_r, K_H_flex=k_f, ratio=k_f / k_r,
                   frac_rigidified=float(fv.rigid.mean()))
        if u_cut is not None:
            # Boltzmann weight (rigid E0) of the rigidified configurations and the estimate with calibrated delta
            w = np.exp(-fv.E0 / kT)
            rec["weight_frac_rigidified"] = float(w[fv.rigid].sum() / w.sum())
            full = FlexibleVext(res, cp, fw, uc, host, T, e_cut=e_n * kT, u_cut=None)
            d_est = np.where(fv.rigid, np.interp(full.u_star_max, uh, ratio) * full.delta, fv.delta)
            e_pair = (fv.E0 + d_est)[fv.pair_cfg]
            v_est = fv._average(e_pair)[0].reshape(res.n) / KB_EV
            rec["K_H_flex_calibrated_estimate"] = henry_from_vext(v_est, res.cellpar, T, mass)
        rec["s"] = round(time.time() - t, 1)
        out["runs"].append(rec)
        print({k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
json.dump(out, open("flex_henry.json", "w"), indent=1, default=float)
