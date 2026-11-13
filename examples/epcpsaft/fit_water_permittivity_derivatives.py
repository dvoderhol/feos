"""Permittivity parameters of water (scheme P: dipole_scaling, correlation_integral_parameter; polarizability_scaling = 0)
fitted to IAPWS R8-97 with temperature derivatives, with the liquid density of ePCP-SAFT (PCP-SAFT water, Esper et al.
2023).

Model (Rueben et al. 2024a, feos PerturbationTheory), at temperature T and number density rho of the EoS:
    eps_r = 1 + 3 y [1 + y + (17/16 I - 1) y^2],   I = 1 + a2 (exp(-y) - 1),
    y = 4 pi / 9 rho [a11mu2 1e-19 / (k_B T) + 3 a12alpha]          (rho in 1/A^3, a11mu2 in D^2, a12alpha in A^3).
At fixed pressure eps_r(T) = eps_r(T, rho(T, p)): the temperature enters explicitly through 1 / T in the dipole term
and implicitly through the density of the EoS.

Fit data (IAPWS R8-97, not experimental data): grid of T from T_MIN to T_MAX (step 5 K) on two isobars, p = 0.1 MPa
(T < 372 K) or 1.5 p_sat (T >= 372 K), and 10 MPa. Quantities, each as relative deviation from IAPWS, derivatives at
constant pressure by central differences (h = 0.5 K) of the model with EoS densities and of IAPWS R8-97:
    "eps"     eps_r
    "d1"      eps_r and d ln(eps_r)/dT
    "d2"      eps_r, d ln(eps_r)/dT and d2(1/eps_r)/dT2
(d ln(eps)/dT enters the Debye-Hueckel and Born enthalpies, d2(1/eps)/dT2 the Born heat capacity.)
Least squares; every quantity has the total weight 1 (residuals divided by sqrt(number of points)).
Additional variant (not available in feos, test of the model form): a2(T) = a2 + b1 (T/T0 - 1) + b2 (T/T0 - 1)^2,
T0 = 298.15 K, fitted to eps_r and d ln(eps_r)/dT ("a2(T) quadratic, eps + d1").
Comparison: the parameters of parameters/epcpsaft/esper2023.json (DDB fit) and, if EPCPSAFT_DDB_PERMITTIVITY is set,
the mean absolute relative deviation from the DDB data (rueben_filter of permittivity.py).
Output: results/water_permittivity_derivatives.csv (parameters and deviations), results/water_permittivity_derivatives.json
(permittivity records), results/water_permittivity_derivatives.png (relative deviations of eps_r, d ln(eps_r)/dT and
d2(1/eps_r)/dT2 along 0.1 MPa / 1.5 p_sat).
usage: python fit_water_permittivity_derivatives.py [--t-min 273.15] [--t-max 573.15]
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import iapws  # noqa: E402
import permittivity as pm  # noqa: E402

WATER = "7732-18-5"
KB = 1.380649e-23
NA = 6.02214076e23
H = 0.5
T0 = 298.15
RESULTS = Path(__file__).resolve().parent / "results"


def eps_model(T, rho, a11mu2, a2, a12alpha=0.0):
    """eps_r of the perturbation-theory model; T in K, rho in 1/A^3"""
    y = 4 * np.pi / 9 * rho * (a11mu2 * 1e-19 / (KB * T) + 3 * a12alpha)
    i = 1 + a2 * (np.exp(-y) - 1)
    return 1 + 3 * y * (1 + y + (17 / 16 * i - 1) * y ** 2)


def quantities(e_minus, e0, e_plus):
    """eps, d ln(eps)/dT, d2(1/eps)/dT2 from values at T - H, T, T + H"""
    return e0, (np.log(e_plus) - np.log(e_minus)) / (2 * H), (1 / e_plus - 2 / e0 + 1 / e_minus) / H ** 2


def grid(t_min, t_max):
    T = np.arange(t_min, t_max + 1e-9, 5.0)
    rows = [(t, iapws.default_pressure(t), "sat") for t in T] + [(t, 1e7, "10MPa") for t in T]
    return pd.DataFrame(rows, columns=["T", "p", "isobar"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-min", type=float, default=273.15)
    ap.add_argument("--t-max", type=float, default=573.15)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    record = pm.esper_records()[WATER]
    current = [record["permittivity_record"]["PerturbationTheory"][n] for n in pm.NAMES]   # a11mu2, a12alpha, a2
    eos = pm.eos(record)
    g = grid(a.t_min, a.t_max)
    # EoS number densities (1/A^3) at T - H, T, T + H; independent of the permittivity parameters
    rho = {k: pm.liquid_density(eos, g["T"] + k * H, g["p"]) * NA * 1e-30 for k in (-1, 0, 1)}
    ref = quantities(*[np.array([iapws.eps_r(t + k * H, p) for t, p in zip(g["T"], g["p"])]) for k in (-1, 0, 1)])

    def model(x):
        def a2(T):
            t = T / T0 - 1
            return x[1] + (x[2] * t + x[3] * t ** 2 if len(x) == 4 else 0.0)
        return quantities(*[eps_model(g["T"] + k * H, rho[k], x[0], a2(g["T"] + k * H)) for k in (-1, 0, 1)])

    def rel(x):
        return [m / r - 1 for m, r in zip(model(x), ref)]

    # consistency of the closed form with feos (State.permittivity) at the current parameters
    feos_eps = pm.permittivity(record, current, g["T"][:3], rho[0][:3] / (NA * 1e-30))
    assert np.allclose(feos_eps, eps_model(g["T"][:3], rho[0][:3], current[0], current[2]), rtol=1e-12)

    fits = {"DDB (esper2023.json)": np.array([current[0], current[2]])}
    for name, used in (("IAPWS eps", 1), ("IAPWS eps + d1", 2), ("IAPWS eps + d1 + d2", 3)):
        res = least_squares(lambda x: np.concatenate([r / np.sqrt(len(r)) for r in rel(x)[:used]]),
                            fits["DDB (esper2023.json)"], bounds=([0, 0], [100, 1]))
        fits[name] = res.x
    res = least_squares(lambda x: np.concatenate([r / np.sqrt(len(r)) for r in rel(x)[:2]]),
                        np.append(fits["DDB (esper2023.json)"], [0.0, 0.0]))
    fits["a2(T) quadratic, eps + d1"] = res.x

    ddb = None
    if os.environ.get("EPCPSAFT_DDB_PERMITTIVITY"):
        d = pm.substance_data(pm.rueben_filter(pm.ddb()), WATER)
        d = d[d["T"] <= a.t_max + 50]
        rho_d = pm.liquid_density(eos, d["T"].values, d["p"].values) * NA * 1e-30
        ddb = (d, rho_d)

    rows = []
    sat = (g.isobar == "sat").values
    low = sat & (g["T"] <= 373.15).values
    for name, x in fits.items():
        r = rel(x)
        row = {"fit": name, "dipole_scaling": x[0], "correlation_integral_parameter": x[1],
               "a2_b1": x[2] if len(x) == 4 else np.nan, "a2_b2": x[3] if len(x) == 4 else np.nan}
        for q, rq in zip(("eps", "dlneps_dT", "d2inveps_dT2"), r):
            row[f"MARD% {q} all"] = 100 * np.mean(np.abs(rq))
            row[f"MARD% {q} 273-373K sat"] = 100 * np.mean(np.abs(rq[low]))
        for Tq in (273.15, 298.15, 323.15, 373.15, 473.15):
            k = np.flatnonzero(sat & np.isclose(g["T"], Tq))
            if len(k):
                row[f"d2inveps % {Tq:.0f}K"] = 100 * r[2][k[0]]
                row[f"dlneps % {Tq:.0f}K"] = 100 * r[1][k[0]]
        if ddb is not None:
            td = ddb[0]["T"].values
            a2d = x[1] + (x[2] * (td / T0 - 1) + x[3] * (td / T0 - 1) ** 2 if len(x) == 4 else 0.0)
            e = eps_model(td, ddb[1], x[0], a2d)
            row["MARD% DDB"] = 100 * np.mean(np.abs(e / ddb[0]["eps"].values - 1))
            row["MAD DDB"] = np.mean(np.abs(e - ddb[0]["eps"].values))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "water_permittivity_derivatives.csv", index=False)
    records = {name: {"PerturbationTheory": {"dipole_scaling": float(x[0]), "polarizability_scaling": 0.0,
                                             "correlation_integral_parameter": float(x[1])}}
               for name, x in fits.items() if len(x) == 2}
    json.dump(records, open(RESULTS / "water_permittivity_derivatives.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(out.set_index("fit").T.round(4).to_string())

    colors = {"DDB (esper2023.json)": "#2a78d6", "IAPWS eps": "#898781", "IAPWS eps + d1": "#eb6834",
              "IAPWS eps + d1 + d2": "#1baf7a", "a2(T) quadratic, eps + d1": "#6250d6"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    titles = [r"$\varepsilon_r$", r"$\partial \ln\varepsilon_r / \partial T$ (DH, Born enthalpy)",
              r"$\partial^2 (1/\varepsilon_r) / \partial T^2$ (Born heat capacity)"]
    T = g["T"][sat]
    for name, x in fits.items():
        for ax, rq in zip(axes, rel(x)):
            ax.plot(T, 100 * rq[sat], color=colors[name], lw=2, ls="--" if name == "IAPWS eps" else "-",
                    label=f"{name}: {x[0]:.3f} D$^2$, a2 = {x[1]:.4f}" + (" + a2(T)" if len(x) == 4 else ""))
    for ax, t in zip(axes, titles):
        ax.axhline(0, color="#0b0b0b", lw=0.8)
        ax.axvspan(273.15, 373.15, color="#e1e0d9", alpha=0.35, lw=0)
        ax.set_title(t, loc="left")
        ax.set_xlabel("T / K (0.1 MPa below 372 K, 1.5 p_sat above)")
    axes[0].set_ylabel("relative deviation from IAPWS R8-97 / %")
    axes[1].set_ylim(-20, 20)
    axes[2].set_ylim(-60, 140)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=3, fontsize=8)
    fig.text(0.01, 0.01, "PCP-SAFT (Esper 2023) water density; derivatives at constant pressure; shaded: temperature "
             "range of the training data", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(RESULTS / "water_permittivity_derivatives.png", dpi=180)


if __name__ == "__main__":
    main()
