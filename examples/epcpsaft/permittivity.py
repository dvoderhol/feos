"""Permittivity parameters of pure solvents for ePCP-SAFT.

Procedure of Rueben et al. (2024a), section 3.1 (data filter, Huber loss, parametrization schemes NP/P/All,
scheme selection), with the liquid density of ePCP-SAFT (identical to PCP-SAFT for systems without ions)
at p = max(1 bar, psat) if no pressure is given (Rueben et al. 2024a, Thermo.py of their repository).
The permittivity is evaluated by feos (`State.permittivity`).

Experimental data: DDB export (licensed, not part of the repository). Path from the environment variable
EPCPSAFT_DDB_PERMITTIVITY.
"""
import json
import os
from pathlib import Path

import feos
import numpy as np
import pandas as pd
import si_units as si
from scipy.optimize import least_squares

PARAMETERS = Path(__file__).resolve().parents[2] / "parameters" / "epcpsaft"
NAMES = ["dipole_scaling", "polarizability_scaling", "correlation_integral_parameter"]
# adjusted entries of (dipole_scaling, polarizability_scaling, correlation_integral_parameter); others = 0
SCHEMES = {"NP": [1], "P": [0, 2], "All": [0, 1, 2]}
# bounds: a2 in [0, 1] (range of the Rueben et al. 2024a database), dipole and polarizability terms in [0, 100]
LOWER = np.array([0.0, 0.0, 0.0])
UPPER = np.array([100.0, 100.0, 1.0])
X0 = np.array([5.0, 5.0, 0.12])
MOL_M3 = si.MOL / si.METER**3


# ---------------------------------------------------------------- parameters
def esper_records(file=PARAMETERS / "esper2023.json"):
    """pure records (dicts in feos JSON format) by CAS number"""
    return {r["identifier"]["cas"]: r for r in json.load(open(file)) if "cas" in r["identifier"]}


def eos(record, permittivity=None):
    """ePCP-SAFT for one pure record (dict); `permittivity` = (dipole_scaling, polarizability_scaling,
    correlation_integral_parameter) replaces the permittivity record"""
    record = dict(record)
    if permittivity is not None:
        record["permittivity_record"] = {"PerturbationTheory": dict(zip(NAMES, map(float, permittivity)))}
    parameters = feos.Parameters.new_pure(feos.PureRecord.from_json_str(json.dumps(record)))
    return feos.EquationOfState.epcpsaft(parameters)


# ---------------------------------------------------------------- data
def ddb(path=None):
    path = path or os.environ.get("EPCPSAFT_DDB_PERMITTIVITY")
    if path is None:
        raise RuntimeError("set EPCPSAFT_DDB_PERMITTIVITY to the DDB permittivity export (tab separated)")
    cols = ["substance_id", "cas", "phase", "temperature", "pressure", "value", "const_prop", "const_value",
            "dipole_moment", "type", "database", "reference_id", "authors", "year"]
    return pd.read_csv(path, sep="\t", low_memory=False, usecols=cols)


# data points removed by decision: (CAS, temperature [K], value, reason)
EXCLUDED = [
    ("141-43-5", 258.15, 3.368, "MEA, Lafontaine 1958: 'Liquid' below the melting point (283.5 K), eps of a solid"),
    ("141-43-5", 263.15, 3.475, "MEA, Lafontaine 1958: 'Liquid' below the melting point (283.5 K), eps of a solid"),
]


# experimental data types of the DDB export ("Experimental": 40 points of one reference, digitized from figures);
# not used: Smoothed, Calculated, Evaluated, Cited, Unreferenced (decision 2026-11-02)
EXPERIMENTAL = ("Experiment", "Experimental")


def rueben_filter(df, max_pressure=None, experimental_only=True, remove_duplicates=True):
    """Rueben et al. (2024a): liquid phase, frequency < 1000 Hz or not given, p < 5 bar or not given.
    Default max_pressure=None: no pressure limit (decision 2026-10-30); max_pressure=5e5: limit of Rueben et al. (2024a).
    experimental_only: only the data types in EXPERIMENTAL; remove_duplicates: see `duplicates` (decision 2026-11-02).
    Points in EXCLUDED are removed."""
    frequency_ok = (df.const_prop == "Frequency [Hz]") & (df.const_value < 1000) | df.const_value.isna()
    keep = (df.phase == "Liquid") & frequency_ok
    if max_pressure is not None:
        keep &= (df.pressure < max_pressure) | df.pressure.isna()
    if experimental_only:
        keep &= df.type.isin(EXPERIMENTAL)
    for cas, T, value, _ in EXCLUDED:
        keep &= ~((df.cas == cas) & np.isclose(df.temperature, T) & np.isclose(df.value, value))
    df = df[keep]
    return df[~duplicates(df)] if remove_duplicates else df


def duplicates(df):
    """Rows that repeat another row (the first one is kept, DDB before ThermoML):
    - same substance, temperature (0.01 K), pressure (100 Pa or both not given) and permittivity (0.001), and
      the same reference or the same first author and year (one publication under two reference ids, mostly DDB and
      ThermoML entries), or
    - identical substance, temperature, pressure (or both not given) and permittivity (exactly equal numbers), from
      any reference (decision 2026-11-02)."""
    key = pd.DataFrame({
        "cas": df.cas, "T": df.temperature.round(2), "p": df.pressure.round(-2).fillna(-1.0), "eps": df.value.round(3),
        "author": df.authors.fillna("").str.split(r"[,;]").str[0].str.strip().str.lower(),
        "year": df.year.astype(str).str[:4], "reference": df.reference_id})
    order = np.argsort((df.database != "DDB").values, kind="stable")
    k = key.iloc[order]
    exact = pd.DataFrame({"cas": df.cas, "T": df.temperature, "p": df.pressure.fillna(-1.0), "eps": df.value}).iloc[order]
    dup = (k.duplicated(["cas", "T", "p", "eps", "reference"]) | k.duplicated(["cas", "T", "p", "eps", "author", "year"])
           | exact.duplicated())
    return pd.Series(dup.values, index=k.index).reindex(df.index)


def substance_data(df, cas):
    """all data points (T [K], p [Pa] or NaN, eps) of one substance"""
    d = df[df.cas == cas][["temperature", "pressure", "value"]].rename(
        columns={"temperature": "T", "pressure": "p", "value": "eps"})
    return d.sort_values("T").reset_index(drop=True)


def liquid_density(eos, T, p):
    """liquid density [mol/m3] at (T, p); p = NaN: p = max(1 bar, psat(T)) of the EoS"""
    out = []
    for t, pp in zip(T, p):
        if np.isnan(pp):
            psat = feos.PhaseEquilibrium.pure(eos, t * si.KELVIN).liquid.pressure() / si.PASCAL
            pp = max(1e5, psat)
        s = feos.State(eos, t * si.KELVIN, pressure=pp * si.PASCAL, density_initialization="liquid")
        out.append(s.density / MOL_M3)
    return np.array(out)


def critical_temperature(eos):
    return feos.State.critical_point(eos).temperature / si.KELVIN


# ---------------------------------------------------------------- model
def permittivity(record, x, T, rho):
    """eps_r from feos for parameters x = (dipole_scaling, polarizability_scaling, correlation_integral_parameter)
    at temperatures T [K] and densities rho [mol/m3]"""
    e = eos(record, x)
    return np.array([feos.State(e, t * si.KELVIN, density=r * MOL_M3).permittivity() for t, r in zip(T, rho)])


# ---------------------------------------------------------------- statistics
def metrics(model, exp):
    """absolute and relative deviations (model - exp)"""
    d = np.asarray(model) - np.asarray(exp)
    r = d / np.asarray(exp)
    return {"n": len(d), "MAD": np.mean(np.abs(d)), "MARD%": 100 * np.mean(np.abs(r)),
            "RMSE": np.sqrt(np.mean(d**2)), "RMSRE%": 100 * np.sqrt(np.mean(r**2)),
            "bias": np.mean(d), "rel_bias%": 100 * np.mean(r),
            "max_AD": np.max(np.abs(d)), "max_RD%": 100 * np.max(np.abs(r))}


def aicc(dev, K):
    """corrected Akaike information criterion, Rueben et al. (2024a) eqs. 14-15"""
    n = len(dev)
    return n * np.log(dev @ dev / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0 else np.nan


def huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


# ---------------------------------------------------------------- fit
def fit(record, T, rho, eps, scheme, fixed=None, x0=None):
    """Fit the parameters of `scheme`. Huber loss with delta = 5 % of the mean eps for n >= 5 data points,
    least squares otherwise (Rueben et al. 2024a). fixed: {index: value} of adjusted parameters held fixed.
    returns dict(x, scheme, metrics, aicc, dev)"""
    T, rho, eps = map(np.asarray, (T, rho, eps))
    fixed = fixed or {}
    idx = [i for i in SCHEMES[scheme] if i not in fixed]
    delta = 0.05 * eps.mean()
    use_huber = len(eps) >= 5

    def full(p):
        x = np.zeros(3)
        x[idx] = p
        for i, v in fixed.items():
            x[i] = v
        return x

    def res(p):
        r = permittivity(record, full(p), T, rho) - eps
        return np.sqrt(2 * huber(r, delta)) * np.sign(r) if use_huber else r

    start = X0 if x0 is None else np.asarray(x0, float)
    p0 = np.clip(start[idx], LOWER[idx] + 1e-6, UPPER[idx] - 1e-6)
    sol = least_squares(res, p0, bounds=(LOWER[idx], UPPER[idx]), x_scale=np.array([1.0, 1.0, 0.01])[idx])
    x = full(sol.x)
    dev = permittivity(record, x, T, rho) - eps
    return {"x": x, "scheme": scheme, "metrics": metrics(dev + eps, eps), "aicc": aicc(dev, len(idx)),
            "dev": dev, "success": sol.success}


def select_scheme(record, T, rho, eps, dipole_moment=None, d_aicc=0.0, a2_one_point=0.12, scheme=None):
    """Scheme selection of Rueben et al. (2024a), Figure 1.
    n >= 5: preselect NP or P by the lower MAD, then All if AICc(All) < AICc(preselected) - d_aicc
            (Rueben: the scheme with the lower AICc, d_aicc = 0).
    n < 5:  P if dipole_moment >= 1 D, else NP; for n = 1 and P, a2 is fixed to a2_one_point (0.12 in Rueben).
    scheme: prescribed scheme (no selection).
    returns (chosen fit, {scheme: fit})"""
    if scheme is not None:
        fixed = {2: a2_one_point} if (len(T) == 1 and scheme == "P") else None
        f = fit(record, T, rho, eps, scheme, fixed=fixed)
        return f, {scheme: f}
    if len(T) >= 5:
        fits = {s: fit(record, T, rho, eps, s) for s in ("NP", "P", "All")}
        pre = "NP" if fits["NP"]["metrics"]["MAD"] < fits["P"]["metrics"]["MAD"] else "P"
        chosen = "All" if fits["All"]["aicc"] < fits[pre]["aicc"] - d_aicc else pre
        return fits[chosen], fits
    if dipole_moment is None or np.isnan(dipole_moment):
        raise ValueError("fewer than 5 data points and no dipole moment: scheme cannot be selected")
    scheme = "P" if dipole_moment >= 1.0 else "NP"
    fixed = {2: a2_one_point} if (len(T) == 1 and scheme == "P") else None
    f = fit(record, T, rho, eps, scheme, fixed=fixed)
    return f, {scheme: f}


def permittivity_record(x):
    return {"PerturbationTheory": dict(zip(NAMES, [round(float(v), 4) for v in x]))}
