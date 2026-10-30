"""Permittivity parameters for all substances of parameters/epcpsaft/esper2023.json with DDB permittivity data.

usage: python fit_solvent_permittivity.py [--processes 32] [--write-parameters]
All data points are used. Writes results/solvent_permittivity.csv (results/ is not tracked: derived from licensed data);
--write-parameters writes the permittivity records of all fitted substances to parameters/epcpsaft/esper2023.json.

Scheme for fewer than 5 data points (Rueben et al. 2024a: dipole moment from DIPPR or ORCA, P if >= 1 D):
dipole moment of the DDB export if given, else the scheme of Rueben's PermittivityParameterDatabasePCSAFT.csv
(path of her repository from EPCPSAFT_RUEBEN_REPO), else the substance is not fitted.
"""
import argparse
import json
import os
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import permittivity as pm

RESULTS = Path(__file__).resolve().parent / "results"
# prescribed schemes (no selection): water P (decision 2026-10-30)
SCHEMES = {"7732-18-5": "P"}


def rueben_pcsaft_database():
    repo = os.environ.get("EPCPSAFT_RUEBEN_REPO")
    if repo is None:
        return None
    return pd.read_csv(Path(repo) / "Parameters" / "PermittivityParameterDatabasePCSAFT.csv").set_index("cas")


def one(args):
    cas, record, data, dipole, rueben = args
    out = {"cas": cas, "name": record["identifier"].get("name"), "n_ddb": len(data)}
    try:
        eos = pm.eos(record)
        tc = pm.critical_temperature(eos)
        data = data[data["T"] < tc]
        out.update(T_c_model=tc, n=len(data))
        if len(data) == 0:
            return dict(out, status="no data below T_c of the model")
        T, p, eps = data["T"].values, data["p"].values, data["eps"].values
        rho = pm.liquid_density(eos, T, p)
        if len(T) < 5 and np.isnan(dipole) and rueben is not None:
            dipole, source = (1.0 if rueben["Scheme"] == "P" else 0.0), "Rueben scheme"
        else:
            source = "DDB dipole moment"
        if cas in SCHEMES:
            out["scheme_source"] = "prescribed"
        elif len(T) < 5:
            out["scheme_source"] = source
        best, fits = pm.select_scheme(record, T, rho, eps, dipole_moment=dipole, scheme=SCHEMES.get(cas))
        out.update(scheme=best["scheme"], **dict(zip(pm.NAMES, best["x"])), **best["metrics"],
                   T_min=T.min(), T_max=T.max(), success=best["success"])
        for s, f in fits.items():
            out[f"MAD_{s}"], out[f"AICc_{s}"] = f["metrics"]["MAD"], f["aicc"]
        if rueben is not None:
            x_r = np.nan_to_num([rueben["a11_mu2"], rueben["a12_alpha"], rueben["a2"]])
            m = pm.metrics(pm.permittivity(record, x_r, T, rho), eps)
            out.update(rueben_scheme=rueben["Scheme"], rueben_dipole_scaling=x_r[0],
                       rueben_polarizability_scaling=x_r[1], rueben_correlation_integral_parameter=x_r[2],
                       rueben_n=rueben["n_points"], rueben_MAD_published=rueben["MAD"],
                       rueben_MAD_here=m["MAD"], rueben_MARD_here=m["MARD%"])
        return dict(out, status="ok")
    except Exception as e:
        return dict(out, status=f"error: {e!r}", traceback=traceback.format_exc(limit=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-parameters", action="store_true")
    ap.add_argument("--processes", type=int, default=os.cpu_count())
    ap.add_argument("--cas", nargs="*", help="only these substances")
    a = ap.parse_args()

    records = pm.esper_records()
    df = pm.rueben_filter(pm.ddb())
    df = df[df.cas.isin(records)]
    rueben = rueben_pcsaft_database()
    jobs = []
    for cas in sorted(df.cas.unique()) if not a.cas else a.cas:
        d = df[df.cas == cas]
        dipole = d.dipole_moment.dropna()
        r = rueben.loc[cas] if rueben is not None and cas in rueben.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        jobs.append((cas, records[cas], pm.substance_data(df, cas),
                     dipole.iloc[0] if len(dipole) else np.nan, r))
    with Pool(a.processes) as pool:
        res = pool.map(one, jobs, chunksize=1)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "solvent_permittivity.csv"
    res = pd.DataFrame(res)
    res.to_csv(out, index=False)
    print(out, res.status.value_counts().to_dict())
    if a.write_parameters:
        write_parameters(res[res.status == "ok"])


def write_parameters(res, file=pm.PARAMETERS / "esper2023.json"):
    """set the permittivity records (4 decimals) of the fitted substances; all other entries unchanged"""
    raw = open(file).read()
    records = json.loads(raw)
    x = {r.cas: [getattr(r, k) for k in pm.NAMES] for r in res.itertuples()}
    for r in records:
        cas = r["identifier"].get("cas")
        if cas in x:
            r["permittivity_record"] = pm.permittivity_record(x[cas])
    open(file, "w").write(json.dumps(records, indent=4, ensure_ascii=True) + ("\n" if raw.endswith("\n") else ""))
    print(f"{file}: permittivity records of {len(x)} substances")


if __name__ == "__main__":
    main()
