"""Pure-water reference: IAPWS-95 (via CoolProp) and IAPWS R8-97 static dielectric constant.

Used to validate the water permittivity of ePCP-SAFT.
"""
import numpy as np
from math import pi
import CoolProp
from CoolProp.CoolProp import PropsSI

_AS = CoolProp.AbstractState("HEOS", "Water")

M_W = 0.018015268  # kg/mol (IAPWS)
R = 8.314462618


def psat(T):
    return PropsSI("P", "T", T, "Q", 0, "Water")


def saturation_temperature(p):
    """saturation temperature of pure water at p [K]"""
    return PropsSI("T", "P", p, "Q", 0, "Water")


def rho_w(T, p):
    """liquid water density [kg/m3] (IAPWS-95; phase forced to liquid so that slightly supercooled states work)"""
    _AS.specify_phase(CoolProp.iphase_liquid)
    _AS.update(CoolProp.PT_INPUTS, p, T)
    _AS.unspecify_phase()
    return _AS.rhomass()


def default_pressure(T):
    """0.1 MPa below 372 K, 1.5 psat above (footnote a of the data specification)"""
    return 1.0e5 if T < 372.0 else 1.5 * psat(T)


def _phi(T, p):
    """fugacity coefficient of pure water vapor at (T, p), p <= psat"""
    _AS.specify_phase(CoolProp.iphase_gas)
    _AS.update(CoolProp.PT_INPUTS, p, T)
    _AS.unspecify_phase()
    return _AS.fugacity_coefficient(0)


def _sat(T):
    """ln f_sat [Pa], psat, liquid molar volume at saturation"""
    ps = psat(T)
    return np.log(ps * _phi(T, ps)), ps, M_W / PropsSI("D", "T", T, "Q", 0, "Water")


def vapor_pressure(T, a_w):
    """pressure over a solution with water activity a_w (only water in the vapor):
    f_vap(T, p) = a_w * f_sat(T) * exp(v_L (p - psat) / RT), solved for p by fixed-point iteration"""
    ln_fsat, ps, vL = _sat(T)
    p = a_w * ps
    for _ in range(50):
        p_new = np.exp(np.log(a_w) + ln_fsat + vL * (p - ps) / (R * T)) / _phi(T, p)
        if abs(p_new - p) < 1e-10 * p:
            break
        p = p_new
    return p_new


def water_activity(T, p):
    """inverse of vapor_pressure: data reduction for real vapor-pressure data"""
    ln_fsat, ps, vL = _sat(T)
    return np.exp(np.log(p * _phi(T, p)) - ln_fsat - vL * (p - ps) / (R * T))


# --- IAPWS R8-97 static dielectric constant (verified against Table 4 of the release)
_N = [0.978224486826, -0.957771379375, 0.237511794148, 0.714692244396, -0.298217036956, -0.108863472196,
      0.949327488264e-1, -0.980469816509e-2, 0.165167634970e-4, 0.937359795772e-4, -0.123179218720e-9]
_I = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10]
_J = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10]


def eps_r(T, p):
    rho = rho_w(T, p) / M_W   # mol/m3 (liquid forced: also slightly supercooled states)
    eps0, alpha, mu, k, NA = 1 / (4e-7 * pi * 299792458.0 ** 2), 1.636e-40, 6.138e-30, 1.380658e-23, 6.0221367e23
    d = rho / (322.0 / 0.018015268)
    g = 1 + sum(n * d ** i * (647.096 / T) ** j for n, i, j in zip(_N, _I, _J)) + 0.196096504426e-2 * d * (T / 228.0 - 1) ** -1.2
    A = NA * mu ** 2 * rho * g / (eps0 * k * T)
    B = NA * alpha * rho / (3 * eps0)
    return (1 + A + 5 * B + np.sqrt(9 + 2 * A + 18 * B + A ** 2 + 10 * A * B + 9 * B ** 2)) / (4 - 4 * B)
