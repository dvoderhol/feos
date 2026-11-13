"""Test of association-dependent permittivity models for water with the density and the association of ePCP-SAFT
(PCP-SAFT water of Esper et al. 2023, 2B association scheme), fitted to IAPWS R8-97 with temperature derivatives.
Water only; the forms below are not part of the feos permittivity models.

Non-bonded site fraction (Wertheim TPT1, 2B, na = nb = 1, as in the association term of feos):
    X_A = (-1 + sqrt(1 + 4 rho Delta)) / (2 rho Delta),   Delta = g_hs(d) kappa_AB sigma^3 (exp(eps_AB / (k_B T)) - 1),
    g_hs(d) = 1 / (1 - eta) + 3/2 eta / (1 - eta)^2 + 1/2 eta^2 / (1 - eta)^3,  d = sigma (1 - 0.12 exp(-3 eps / (k_B T))),
    rho: number density of molecules; P = 1 - X_A: bonded site fraction.
Forms (eps_PT(T, rho; a11mu2, a2): perturbation theory of Rueben et al. 2024a as in fit_water_permittivity_derivatives.py,
a12alpha = 0):
    "PT"                  eps_PT(a11mu2, a2)                                                          2 parameters
    "PT, X_A^c"           eps_PT(a11mu2 X_A^c, a2)                                                    3 parameters
    "PT, rational"        eps_PT(a11mu2 [1 + c P / (delta + (1 - delta) X_A)], a2),  0 < delta <= 1     4 parameters
    "KF, g(X_A)"          Kirkwood-Froehlich equation (Hasted form) with the Kirkwood factor of Maribo-Mogensen et al.
                          (J. Phys. Chem. B 117 (2013) 3389, Eq. 5.22 of the thesis, DTU 2014) for one molecule type:
                          (2 eps + eps_inf)(eps - eps_inf) / (eps (eps_inf + 2)^2) = rho g mu0^2 / (9 eps0 k_B T),
                          g = 1 + zcg P / (1 + P ct)  (zcg = z cos(gamma), ct = cos(theta)),
                          (eps_inf - 1) / (eps_inf + 2) = rho alpha0 / (3 eps0),
                          mu0 = 1.855 D and alpha0 = 1.613e-40 C^2 m^2 / J fixed (Maribo-Mogensen 2013)     2 parameters
Fit data (IAPWS R8-97): T grid (5 K) on the isobars 0.1 MPa (T < 372 K) or 1.5 p_sat (T >= 372 K) and 10 MPa;
relative deviations of eps_r, d ln(eps_r)/dT and d2(1/eps_r)/dT2 at constant pressure (central differences, h = 0.5 K;
model with EoS densities), every quantity with total weight 1. Fits:
    "eps + d1, 273-373 K"         eps_r and d ln(eps_r)/dT for 273.15 <= T <= 373.15 K
    "eps + d1 + d2, 273-573 K"    all three quantities for 273.15 <= T <= 573.15 K
Reported in addition (not fitted): d ln(eps_r)/dp at constant temperature (central differences, h = 0.5 MPa) on the first
isobar, which enters the apparent molar volume through the Debye-Hueckel and Born terms.
Check: X_A against the association contribution of feos, A_assoc / (N k_B T) = 2 ln X_A - X_A + 1.
Comparison: if EPCPSAFT_DDB_PERMITTIVITY is set, mean absolute relative deviation from the DDB data (rueben_filter of
permittivity.py, T <= 623.15 K).
Output: results/water_permittivity_association.csv (parameters and deviations), results/water_permittivity_association.png
(relative deviations along 0.1 MPa / 1.5 p_sat).
Threads: RAYON_NUM_THREADS.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import si_units as si  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import feos  # noqa: E402
import iapws  # noqa: E402
import permittivity as pm  # noqa: E402

WATER = "7732-18-5"
KB = 1.380649e-23
NA = 6.02214076e23
EPS0 = 8.8541878128e-12
DEBYE = 3.33564e-30
MU0 = 1.855 * DEBYE
ALPHA0 = 1.613e-40
H = 0.5
HP = 5e5
RESULTS = Path(__file__).resolve().parent / "results"

RECORD = pm.esper_records()[WATER]
EOS = pm.eos(RECORD)
SITE = RECORD["association_sites"][0]


def non_bonded_fraction(T, rho):
    """X_A of the 2B model; T in K, rho in molecules / A^3"""
    d = RECORD["sigma"] * (1 - 0.12 * np.exp(-3 * RECORD["epsilon_k"] / T))
    eta = np.pi / 6 * rho * RECORD["m"] * d ** 3
    g = 1 / (1 - eta) + 1.5 * eta / (1 - eta) ** 2 + 0.5 * eta ** 2 / (1 - eta) ** 3
    rd = rho * g * SITE["kappa_ab"] * RECORD["sigma"] ** 3 * np.expm1(SITE["epsilon_k_ab"] / T)
    return (-1 + np.sqrt(1 + 4 * rd)) / (2 * rd)


def eps_pt(T, rho, a11mu2, a2):
    y = 4 * np.pi / 9 * rho * a11mu2 * 1e-19 / (KB * T)
    i = 1 + a2 * (np.exp(-y) - 1)
    return 1 + 3 * y * (1 + y + (17 / 16 * i - 1) * y ** 2)


def eps_kf(T, rho, zcg, ct):
    n = rho * 1e30
    x = non_bonded_fraction(T, rho)
    g = 1 + zcg * (1 - x) / (1 + (1 - x) * ct)
    c = n * ALPHA0 / (3 * EPS0)
    e_inf = (1 + 2 * c) / (1 - c)
    a = n * g * MU0 ** 2 / (9 * EPS0 * KB * T) * (e_inf + 2) ** 2
    return ((e_inf + a) + np.sqrt((e_inf + a) ** 2 + 8 * e_inf ** 2)) / 4


def eps_form(form, x, T, rho):
    if form == "PT":
        return eps_pt(T, rho, x[0], x[1])
    xa = non_bonded_fraction(T, rho)
    if form == "PT, X_A^c":
        return eps_pt(T, rho, x[0] * xa ** x[2], x[1])
    if form == "PT, rational":
        return eps_pt(T, rho, x[0] * (1 + x[2] * (1 - xa) / (x[3] + (1 - x[3]) * xa)), x[1])
    if form == "KF, g(X_A)":
        return eps_kf(T, rho, x[0], x[1])
    raise ValueError(form)


FORMS = {   # start values, lower bounds, upper bounds
    "PT": ([5.26, 0.13], [0, 0], [100, 1]),
    "PT, X_A^c": ([5.26, 0.13, 0.0], [0, 0, -5], [100, 1, 5]),
    "PT, rational": ([5.26, 0.13, 0.0, 0.5], [0, 0, -50, 1e-3], [100, 1, 50, 1]),
    "KF, g(X_A)": ([2.0, 0.0], [-20, -0.999], [20, 0.999]),
}
FITS = {"eps + d1, 273-373 K": (2, 273.15, 373.15), "eps + d1 + d2, 273-573 K": (3, 273.15, 573.15)}
QUANTITIES = ("eps", "dlneps_dT", "d2inveps_dT2", "dlneps_dp")


def number_density(T, p):
    return pm.liquid_density(EOS, T, p) * NA * 1e-30


def main():
    RESULTS.mkdir(exist_ok=True)
    T = np.arange(273.15, 573.16, 5.0)
    g = pd.DataFrame([(t, iapws.default_pressure(t), "sat") for t in T] + [(t, 1e7, "10MPa") for t in T],
                     columns=["T", "p", "isobar"])
    Tg, pg = g["T"].values, g["p"].values
    rho_t = {k: number_density(Tg + k * H, pg) for k in (-1, 0, 1)}
    rho_p = {k: number_density(Tg, pg + k * HP) for k in (-1, 1)}

    # X_A against the association contribution of feos
    for t, r in zip(Tg[:: 20], rho_t[0][:: 20]):
        s = feos.State(EOS, t * si.KELVIN, density=r / (NA * 1e-30) * si.MOL / si.METER ** 3)
        assoc = dict(s.residual_molar_helmholtz_energy_contributions())["Association"]
        xa = non_bonded_fraction(t, r)
        a_feos = assoc / (si.JOULE / si.MOL) / (KB * NA * t)
        assert abs(a_feos - (2 * np.log(xa) - xa + 1)) < 1e-8 * abs(a_feos), (t, a_feos, xa)

    def iapws_eps(dt, dp):
        return np.array([iapws.eps_r(t + dt, p + dp) for t, p in zip(Tg, pg)])

    ref = [iapws_eps(0, 0)]
    em, ep = iapws_eps(-H, 0), iapws_eps(H, 0)
    ref += [(np.log(ep) - np.log(em)) / (2 * H), (1 / ep - 2 / ref[0] + 1 / em) / H ** 2]
    ref += [(np.log(iapws_eps(0, HP)) - np.log(iapws_eps(0, -HP))) / (2 * HP)]

    def model(form, x):
        e = {k: eps_form(form, x, Tg + k * H, rho_t[k]) for k in (-1, 0, 1)}
        out = [e[0], (np.log(e[1]) - np.log(e[-1])) / (2 * H), (1 / e[1] - 2 / e[0] + 1 / e[-1]) / H ** 2]
        out += [(np.log(eps_form(form, x, Tg, rho_p[1])) - np.log(eps_form(form, x, Tg, rho_p[-1]))) / (2 * HP)]
        return out

    def rel(form, x):
        return [m / r - 1 for m, r in zip(model(form, x), ref)]

    current = RECORD["permittivity_record"]["PerturbationTheory"]
    results = {("PT", "DDB (esper2023.json)"): np.array([current["dipole_scaling"],
                                                          current["correlation_integral_parameter"]])}
    for form, (x0, lo, hi) in FORMS.items():
        for fit, (nq, t_lo, t_hi) in FITS.items():
            mask = (Tg >= t_lo - 1e-9) & (Tg <= t_hi + 1e-9)

            def res(x):
                r = rel(form, x)[:nq]
                return np.concatenate([q[mask] / np.sqrt(mask.sum()) for q in r])
            best = None
            for start in ([x0] + ([[x0[0], x0[1], 0.3, 0.1], [x0[0], x0[1], -0.3, 0.1]] if form == "PT, rational" else [])):
                try:
                    sol = least_squares(res, start, bounds=(lo, hi))
                except ValueError:
                    continue
                if best is None or sol.cost < best.cost:
                    best = sol
            results[(form, fit)] = best.x

    ddb = None
    if os.environ.get("EPCPSAFT_DDB_PERMITTIVITY"):
        d = pm.substance_data(pm.rueben_filter(pm.ddb()), WATER)
        d = d[d["T"] <= 623.15]
        ddb = (d, number_density(d["T"].values, d["p"].values))

    sat = g.isobar.values == "sat"
    low = sat & (Tg <= 373.15 + 1e-9)
    high = sat & (Tg > 373.15 + 1e-9)
    rows = []
    for (form, fit), x in results.items():
        r = rel(form, x)
        row = {"form": form, "fit": fit, "parameters": " ".join(f"{v:.6g}" for v in x)}
        for q, rq in zip(QUANTITIES, r):
            row[f"MARD% {q} 273-373K"] = 100 * np.mean(np.abs(rq[low]))
            row[f"MARD% {q} 373-573K"] = 100 * np.mean(np.abs(rq[high]))
        for Tq in (273.15, 298.15, 373.15, 473.15):
            k = np.flatnonzero(sat & np.isclose(Tg, Tq))[0]
            for q, rq in zip(QUANTITIES[1:], r[1:]):
                row[f"{q} % {Tq:.0f}K"] = 100 * rq[k]
        if ddb is not None:
            e = eps_form(form, x, ddb[0]["T"].values, ddb[1])
            row["MARD% DDB"] = 100 * np.mean(np.abs(e / ddb[0]["eps"].values - 1))
            row["MAD DDB"] = np.mean(np.abs(e - ddb[0]["eps"].values))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "water_permittivity_association.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.set_index(["form", "fit"]).T.to_string(float_format=lambda v: f"{v:.3g}"))
    xa = non_bonded_fraction(Tg[sat], rho_t[0][sat])
    print("X_A at", ", ".join(f"{t:.0f} K: {v:.4f}" for t, v in zip(Tg[sat][::10], xa[::10])))

    colors = {"PT": "#2a78d6", "PT, X_A^c": "#eb6834", "PT, rational": "#1baf7a", "KF, g(X_A)": "#6250d6"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9))
    titles = [r"$\varepsilon_r$", r"$\partial \ln\varepsilon_r / \partial T$",
              r"$\partial^2 (1/\varepsilon_r) / \partial T^2$", r"$\partial \ln\varepsilon_r / \partial p$ (not fitted)"]
    for (form, fit), x in results.items():
        ls = ":" if fit.startswith("DDB") else ("-" if fit.startswith("eps + d1,") else "--")
        for ax, rq in zip(axes, rel(form, x)):
            ax.plot(Tg[sat], 100 * rq[sat], color=colors[form], lw=1.8, ls=ls, label=f"{form}; {fit}")
    for ax, t, lim in zip(axes, titles, (5, 30, 100, 60)):
        ax.axhline(0, color="#0b0b0b", lw=0.8)
        ax.axvspan(273.15, 373.15, color="#e1e0d9", alpha=0.35, lw=0)
        ax.set_ylim(-lim, lim)
        ax.set_title(t, loc="left")
        ax.set_xlabel("T / K (0.1 MPa below 372 K, 1.5 p_sat above)")
    axes[0].set_ylabel("relative deviation from IAPWS R8-97 / %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=3, fontsize=7.5)
    fig.text(0.01, 0.01, "PCP-SAFT water (Esper 2023): density and X_A; solid: fit to eps + d1 on 273-373 K, dashed: fit "
             "to eps + d1 + d2 on 273-573 K, dotted: DDB parameters; shaded: temperature range of the training data; "
             "axes clipped", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.80))
    fig.savefig(RESULTS / "water_permittivity_association.png", dpi=180)


if __name__ == "__main__":
    main()
