"""Fit a spherical (m = 1) quadrupolar PC-SAFT parameter set for CO2 to the reference equation of state
(Span-Wagner, CoolProp parameters shipped with feos): vapour pressure and saturated liquid density, 220-295 K,
with the quadrupole moment fixed at Q = 4.4 DA (the value of Gross 2005, AIChE J. 51, 2556).

Purpose (fluid-model test, 2026-11-18): an m = 1 fluid model with quadrupolar CO2-CO2 interactions, so that the
molecule-centred MLIP V_ext still enters exactly (K_H = <exp(-V/kT)>) and the effect of the fluid model on the
isotherm can be separated from the effect of the V_ext mapping onto chain segments (Gross 2005, m = 1.513).
Same data, temperatures and objective as fit_co2_m1.py. Writes co2_pcsaft_m1_q4.4.json (feos PureRecord format).
"""
import json
from pathlib import Path

import numpy as np
import si_units as si
import feos
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
MW = 44.0098
Q = 4.4
ref = feos.EquationOfState.multiparameter(feos.Parameters.from_json(
    ["CarbonDioxide"], str(Path.home() / "feos/parameters/multiparameter/coolprop.json")))
T = np.linspace(220.0, 295.0, 16)
ref_vle = [feos.PhaseEquilibrium.pure(ref, t * si.KELVIN) for t in T]
p_ref = np.array([float(v.vapor.pressure() / si.BAR) for v in ref_vle])
rho_ref = np.array([float(v.liquid.density / (si.MOL / si.METER ** 3)) for v in ref_vle])


def pcsaft(sigma, eps_k, m=1.0, q=Q):
    rec = feos.PureRecord(feos.Identifier(name="carbon dioxide", cas="124-38-9"), molarweight=MW,
                          m=m, sigma=sigma, epsilon_k=eps_k, q=q)
    return feos.EquationOfState.pcsaft(feos.Parameters.new_pure(rec))


def residuals(x):
    eos = pcsaft(*x)
    r = []
    for t, p0, d0 in zip(T, p_ref, rho_ref):
        try:
            v = feos.PhaseEquilibrium.pure(eos, t * si.KELVIN)
            r += [float(v.vapor.pressure() / si.BAR) / p0 - 1, float(v.liquid.density / (si.MOL / si.METER ** 3)) / d0 - 1]
        except Exception:  # noqa: BLE001
            r += [1.0, 1.0]
    return np.array(r)


fit = least_squares(residuals, x0=[3.6, 200.0], x_scale=[0.1, 10.0])
sigma, eps_k = fit.x
res = residuals(fit.x).reshape(-1, 2)
aard = (100 * np.abs(res[:, 0]).mean(), 100 * np.abs(res[:, 1]).mean())
print(f"m = 1, Q = {Q} DA fit: sigma = {sigma:.5f} A, epsilon_k = {eps_k:.4f} K; AARD p_sat {aard[0]:.2f} %, "
      f"rho_liq {aard[1]:.2f} % (220-295 K), cost {fit.cost:.3e}, status {fit.status}")
# sanity: the fit must not sit at the initial guess or on the failure plateau (r = 1)
assert fit.status > 0 and np.all(np.abs(res) < 0.5), "fit failed"

g2005 = feos.EquationOfState.pcsaft(feos.Parameters.from_json(
    ["carbon dioxide"], str(Path.home() / "feos/parameters/pcsaft/gross2005_literature.json")))
m1 = feos.EquationOfState.pcsaft(feos.Parameters.from_json(["carbon dioxide"], str(HERE / "co2_pcsaft_m1.json")))
m1q = pcsaft(sigma, eps_k)
print("  p/bar   phi(Span-Wagner)  phi(m=1)  phi(m=1,Q)  phi(gross2005)")
for p in [0.001, 0.1, 1, 5, 10, 20, 40]:
    phis = [float(np.exp(feos.State(e, 298 * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor").ln_phi()[0]))
            for e in (ref, m1, m1q, g2005)]
    print(f"  {p:6.3f}  {phis[0]:.5f}          {phis[1]:.5f}   {phis[2]:.5f}     {phis[3]:.5f}")
json.dump([{"identifier": {"name": "carbon dioxide", "cas": "124-38-9"}, "molarweight": MW, "m": 1.0,
            "sigma": round(float(sigma), 6), "epsilon_k": round(float(eps_k), 5), "q": Q,
            "comment": (f"m = 1, Q = {Q} DA (fixed, Gross 2005) fit to Span-Wagner (CoolProp) p_sat and rho_liq, "
                        f"220-295 K; AARD p_sat {aard[0]:.2f} %, rho_liq {aard[1]:.2f} % "
                        "(feos_mlip/parameters/fit_co2_m1_q.py)")}],
          open(HERE / "co2_pcsaft_m1_q4.4.json", "w"), indent=1)
