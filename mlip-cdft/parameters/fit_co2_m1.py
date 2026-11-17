"""Fit a spherical (m = 1) PC-SAFT parameter set for CO2 to the reference equation of state (Span-Wagner,
CoolProp parameters shipped with feos): vapour pressure and saturated liquid density, 220-295 K.

Rationale: the MLIP external potential is an orientation-averaged free energy of the whole molecule at its
centre (C atom). With m = 1 feos applies it to one segment = one molecule, so it enters exactly
(K_H = <exp(-V/kT)>), whereas for m > 1 feos spreads it along a chain (V/m per segment + bond integrals).
Writes co2_pcsaft_m1.json (feos PureRecord format).
"""
import json
from pathlib import Path

import numpy as np
import si_units as si
import feos
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
MW = 44.0098
ref = feos.EquationOfState.multiparameter(feos.Parameters.from_json(
    ["CarbonDioxide"], str(Path.home() / "feos/parameters/multiparameter/coolprop.json")))
T = np.linspace(220.0, 295.0, 16)
ref_vle = [feos.PhaseEquilibrium.pure(ref, t * si.KELVIN) for t in T]
p_ref = np.array([float(v.vapor.pressure() / si.BAR) for v in ref_vle])
rho_ref = np.array([float(v.liquid.density / (si.MOL / si.METER ** 3)) for v in ref_vle])


def pcsaft(sigma, eps_k, m=1.0):
    rec = feos.PureRecord(feos.Identifier(name="carbon dioxide", cas="124-38-9"), molarweight=MW,
                          m=m, sigma=sigma, epsilon_k=eps_k)
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


fit = least_squares(residuals, x0=[3.7, 230.0], x_scale=[0.1, 10.0])
sigma, eps_k = fit.x
res = residuals(fit.x).reshape(-1, 2)
print(f"m = 1 fit: sigma = {sigma:.5f} A, epsilon_k = {eps_k:.4f} K; AARD p_sat {100*np.abs(res[:,0]).mean():.2f} %, "
      f"rho_liq {100*np.abs(res[:,1]).mean():.2f} % (220-295 K)")

# reservoir check at the adsorption temperature: fugacity coefficient of the gas, 298 K, 0.001-40 bar
g2001 = feos.EquationOfState.pcsaft(feos.Parameters.from_json(["carbon dioxide"], str(Path.home() / "feos/parameters/pcsaft/gross2001.json")))
m1 = pcsaft(sigma, eps_k)
print("  p/bar   phi(Span-Wagner)  phi(m=1 fit)  phi(gross2001)")
for p in [0.001, 0.1, 1, 5, 10, 20, 40]:
    phis = [float(np.exp(feos.State(e, 298 * si.KELVIN, pressure=p * si.BAR, density_initialization="vapor").ln_phi()[0]))
            for e in (ref, m1, g2001)]
    print(f"  {p:6.3f}  {phis[0]:.5f}          {phis[1]:.5f}       {phis[2]:.5f}")
json.dump([{"identifier": {"name": "carbon dioxide", "cas": "124-38-9"}, "molarweight": MW, "m": 1.0,
            "sigma": round(float(sigma), 6), "epsilon_k": round(float(eps_k), 5),
            "comment": "m = 1 fit to Span-Wagner (CoolProp) p_sat and rho_liq, 220-295 K (feos_mlip/parameters/fit_co2_m1.py)"}],
          open(HERE / "co2_pcsaft_m1.json", "w"), indent=1)
