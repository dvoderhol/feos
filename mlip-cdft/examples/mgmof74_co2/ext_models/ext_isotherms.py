"""cDFT CO2 isotherms in Mg-MOF-74 at 298 K with V_ext from published NequIP potentials (make_vext_nequip.py).

usage: python ext_isotherms.py <model key> <tag> <guard> [guard ...]     (guards: raw, paper, u1.8, paper+Eguest)
"paper+Eguest": V_ext + E_guest, where E_guest is the model energy of an isolated CO2. This reproduces the energy
convention of Goeminne's gcmc.py, whose insertion energy E(N+1) - E(N) does not subtract E_guest (the models were
trained on intermolecular energies, but predict E_guest != 0 for an isolated CO2). It is a replication of their
protocol, not a physical correction.
Same cDFT as the uMOF-MH baseline (review_followup.py): PC-SAFT m = 1 CO2 (parameters/co2_pcsaft_m1.json),
AntiSymWhiteBear FMT, nested feos grid n_feos = 2 x n_MLIP, adaptive adsorption branch, absolute adsorption.
Pressures 1e-4 .. 50 bar (63 points; Goeminne's GCMC goes to 50 bar). Writes iso_<key>_<tag>_<guard>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import feos

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import MLIPPore3D, _Pore3DBase, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
HERE = Path(__file__).resolve().parent
key, tag, guards = sys.argv[1], sys.argv[2], sys.argv[3:]
pressures = np.geomspace(1e-4, 50, 63) if tag != "smoke" else np.geomspace(1e-4, 50, 7)
func = pcsaft(["carbon dioxide"], PARS, feos.FMTVersion.AntiSymWhiteBear)
for g in guards:
    shift_eV = 0.0
    gfile = g
    if g.endswith("+Eguest"):
        gfile = g.split("+")[0]
    res = VextResult.load(HERE / f"vext_{key}_{tag}_{gfile}.npz")
    mass = res.info["framework_mass_amu"]
    n_nested = [2 * k for k in res.n]
    pore = MLIPPore3D(res, T, n_grid=n_nested, framework_mass_amu=mass, m=1.0)
    if g.endswith("+Eguest"):
        shift_eV = float(res.info["E_guest"])
        pore = _Pore3DBase(pore.cellpar, pore.vext_k + shift_eV / 8.617333262e-5, mass, T, m=1.0)
    t = time.time()
    iso = adaptive_isotherm(func, pore, T, pressures)
    wall = time.time() - t
    kh = pore.henry_coefficient(T)
    p, n = iso["pressure_bar"], iso["mol_per_kg"]
    print(f"{key} {g}: K_H {kh:.4e} mol/(kg Pa), {len(p)} points to {p[-1]:.3g} bar (end {iso['p_end']}), {wall:.0f} s; "
          f"dilute check n/(K_H p) at {p[0]:.0e} bar = {n[0] / (kh * p[0] * 1e5):.4f}", flush=True)
    for q, x in list(zip(p, n))[::max(1, len(p) // 12)]:
        print(f"   {q:9.4g} bar  {x:7.3f} mmol/g", flush=True)
    json.dump({"T_K": T, "key": key, "guard": g, "fluid": "PC-SAFT m=1 (co2_pcsaft_m1.json)", "fmt": "AntiSymWhiteBear",
               "grid_feos": n_nested, "vext_info": res.info, "V_shift_eV": shift_eV, "wall_s": wall, "K_H": kh,
               "pressure_bar": p.tolist(), "mmol_per_g": n.tolist(), "p_end": iso["p_end"]},
              open(HERE / f"iso_{key}_{tag}_{g}.json", "w"), indent=1, default=str)
