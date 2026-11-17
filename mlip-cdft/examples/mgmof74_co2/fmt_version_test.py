"""Does the FMT variant fix the end of the adsorption branch? White Bear (feos default) vs AntiSymWhiteBear
(correct zero-dimensional limit, clamps |n2v| <= n2), adaptive pressure continuation to the highest pressure,
plus the fixed-N trace for CO2 + MLIP (is the van der Waals loop still there?)."""
import json
import sys
import time
import numpy as np
import feos
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, _Pore3DBase, adaptive_isotherm, canonical_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz")
v_ch4 = np.load("../mgmof74_ch4/v_lj_uff_0.3A.npy")
pores = {"CH4 + UFF": (lambda: _Pore3DBase(res.cellpar, np.where(v_ch4 < 50 * T, v_ch4, np.inf), mass),
                       ["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json", np.geomspace(0.1, 100, 31)),
         "CO2 + UFF": (lambda: LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T),
                       ["carbon dioxide"], PARS, np.geomspace(1e-4, 40, 36)),
         "CO2 + MLIP": (lambda: MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0),
                        ["carbon dioxide"], PARS, np.geomspace(1e-4, 40, 36))}
out = {}
for fmt in ["AntiSymWhiteBear", "WhiteBear"]:
    for name, (mk, comp, pfile, p) in pores.items():
        if fmt == "WhiteBear" and name != "CO2 + MLIP":
            continue   # White Bear results for the others are known (branch ends at 10.7 / 0.65 bar)
        pore, func, t = mk(), pcsaft(comp, pfile, getattr(feos.FMTVersion, fmt)), time.time()
        iso = adaptive_isotherm(func, pore, T, p)
        print(f"--- {name}, {fmt}: reached {iso['pressure_bar'][-1]:.4g} bar (branch end {iso['p_end']}), "
              f"{time.time()-t:.0f}s", flush=True)
        for pb, n in list(zip(iso["pressure_bar"], iso["mol_per_kg"]))[::3]:
            print(f"   {pb:10.4g} bar  {n:7.3f} mmol/g", flush=True)
        out[f"{name}|{fmt}"] = {"pressure_bar": iso["pressure_bar"].tolist(), "mmol_per_g": iso["mol_per_kg"].tolist(),
                                "p_end": iso["p_end"]}
        if name == "CO2 + MLIP":
            can = canonical_isotherm(func, pore, T, np.arange(0.25, 30, 0.25))
            pp = can["pressure_bar"]
            print(f"   fixed-N trace: N up to {can['molecules_per_cell'].max() if len(pp) else 0:.2f}/cell, "
                  f"p(N) monotonic: {bool(np.all(np.diff(pp) > 0))}, stop: {can['failed']}", flush=True)
            out[f"{name}|{fmt}|canonical"] = {"N": can["molecules_per_cell"].tolist(), "pressure_bar": pp.tolist(),
                                               "mmol_per_g": can["mol_per_kg"].tolist()}
json.dump(out, open("fmt_version_test.json", "w"), indent=1)
