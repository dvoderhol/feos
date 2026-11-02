"""Permittivity parameters for all substances of parameters/epcpsaft/esper2023.json with DDB permittivity data.

usage: python fit_solvent_permittivity.py [--processes 32] [--max-pressure none|5e5] [--write-parameters]
All data points are fitted (default, decision 2026-10-30); with --max-pressure only points with p < max_pressure (Pa;
Rueben et al. 2024a: 5 bar) or without pressure.
Errors are reported for all liquid points below 5 bar (columns *_low) and at or above 5 bar (columns *_high).
Writes results/solvent_permittivity[_p<max_pressure>].csv (results/ is not tracked: derived from licensed data);
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
    cas, record, data_all, dipole, rueben, max_pressure = args
    out = {"cas": cas, "name": record["identifier"].get("name")}
    try:
        eos = pm.eos(record)
        tc = pm.critical_temperature(eos)
        data_all = data_all[data_all["T"] < tc].reset_index(drop=True)
        rho_all = pm.liquid_density(eos, data_all["T"].values, data_all["p"].values)
        fitted = (data_all["p"] < max_pressure).values | data_all["p"].isna().values if max_pressure else np.ones(len(data_all), bool)
        low = (data_all["p"] < 5e5).values | data_all["p"].isna().values
        data, rho = data_all[fitted], rho_all[fitted]
        out.update(T_c_model=tc, n=len(data), n_low=low.sum(), n_high=(~low).sum(),
                   p_max=data["p"].max() if data["p"].notna().any() else np.nan)
        if len(data) == 0:
            return dict(out, status="no data below T_c of the model")
        T, p, eps = data["T"].values, data["p"].values, data["eps"].values
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
        eps_all = pm.permittivity(record, best["x"], data_all["T"].values, rho_all)
        for key, mask in (("low", low), ("high", ~low)):
            if mask.any():
                m = pm.metrics(eps_all[mask], data_all["eps"].values[mask])
                out.update({f"{k}_{key}": m[k] for k in ("MAD", "MARD%", "RMSE", "RMSRE%", "bias", "rel_bias%", "max_AD", "max_RD%")})
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
    ap.add_argument("--max-pressure", default="none", help="Pa, or 'none' for no limit (default)")
    a = ap.parse_args()
    max_pressure = None if a.max_pressure.lower() == "none" else float(a.max_pressure)

    records = pm.esper_records()
    df = pm.rueben_filter(pm.ddb(), max_pressure=None)    # all liquid points; the fit uses p < max_pressure
    df = df[df.cas.isin(records)]
    fit_ok = df.pressure.isna() | (df.pressure < max_pressure) if max_pressure else df.pressure.notna() | df.pressure.isna()
    df = df[df.cas.isin(df[fit_ok].cas)]                   # substances with at least one point to fit
    rueben = rueben_pcsaft_database()
    jobs = []
    for cas in sorted(df.cas.unique()) if not a.cas else a.cas:
        d = df[df.cas == cas]
        dipole = d.dipole_moment.dropna()
        r = rueben.loc[cas] if rueben is not None and cas in rueben.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        jobs.append((cas, records[cas], pm.substance_data(df, cas),
                     dipole.iloc[0] if len(dipole) else np.nan, r, max_pressure))
    with Pool(a.processes) as pool:
        res = pool.map(one, jobs, chunksize=1)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("solvent_permittivity.csv" if max_pressure is None else f"solvent_permittivity_p{a.max_pressure}.csv")
    res = pd.DataFrame(res)
    res.to_csv(out, index=False)
    print(out, res.status.value_counts().to_dict())
    if a.write_parameters:
        write_parameters(res[res.status == "ok"])


def write_parameters(res, file=pm.PARAMETERS / "esper2023.json"):
    """set the permittivity records (4 decimals) of the fitted substances and remove the permittivity records of all
    other substances (all permittivity records of the file are results of this fit); all other entries unchanged"""
    raw = open(file).read()
    records = json.loads(raw)
    x = {r.cas: [getattr(r, k) for k in pm.NAMES] for r in res.itertuples()}
    for r in records:
        cas = r["identifier"].get("cas")
        if cas in x:
            r["permittivity_record"] = pm.permittivity_record(x[cas])
        else:
            r.pop("permittivity_record", None)
    open(file, "w").write(json.dumps(records, indent=4, ensure_ascii=True) + ("\n" if raw.endswith("\n") else ""))
    print(f"{file}: permittivity records of {len(x)} substances")


if __name__ == "__main__":
    main()
