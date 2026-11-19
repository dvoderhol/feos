"""Goeminne et al. 2023 rigid GCMC isotherms (NequIP PBE-D3(BJ)/TS/MBD, 298 K) from their raw uptake time series
(Zenodo 7904959, data/gcmc_results/Mg-MOF-74/results_*/uptake_<p>bar.npy: CO2 per 54-atom cell per MC iteration,
2.5e6 iterations). Absolute adsorption; mmol/g with the cell mass of their framework (728.1 g/mol).
Their equilibration cut is not stated: means after discarding 20 % and 50 % are both reported; error = standard error
of 10 blocks of the second half. Writes goeminne_gcmc_reference.json."""
import io, json, re, sys
import numpy as np
sys.path.insert(0, "/local/home/dvoderhol/umof_models_ext")
from zenodo_zip import open_zip
from ase.io import read
M = float(read("/local/home/dvoderhol/umof_models_ext/goeminne2023/structures/MgMOF74_goeminne_PBED3_framework.cif").get_masses().sum())
z = open_zip(); out = {"cell_mass_g_mol": M}
for fn in ["PBED3", "PBETS", "PBEMBD"]:
    rows = []
    for name in sorted(n for n in z.namelist() if f"results_{fn}/uptake_" in n):
        p = float(re.search(r"uptake_([0-9.]+)bar", name).group(1))
        a = np.load(io.BytesIO(z.read(name))).astype(float); n = len(a); h = a[n // 2:]
        blocks = np.array([b.mean() for b in np.array_split(h, 10)])
        rows.append(dict(p=p, n_iter=n, N_cell_after20=float(a[n // 5:].mean()), N_cell_after50=float(h.mean()),
                         err_N=float(blocks.std(ddof=1) / np.sqrt(10)),
                         mmol_g_after20=float(a[n // 5:].mean() / M * 1000), mmol_g_after50=float(h.mean() / M * 1000),
                         err_mmol_g=float(blocks.std(ddof=1) / np.sqrt(10) / M * 1000)))
    rows.sort(key=lambda r: r["p"]); out[fn] = rows
    print(fn); [print(f"  {r['p']:9.4g} bar  {r['mmol_g_after50']:7.3f} +- {r['err_mmol_g']:.3f}  (after 20 %: {r['mmol_g_after20']:.3f})  N/cell {r['N_cell_after50']:.3f}") for r in rows]
json.dump(out, open("goeminne_gcmc_reference.json", "w"), indent=1)
