"""Experimental (or dummy) data of aqueous electrolyte solutions for the ion-parameter fit of ePCP-SAFT.

Data format: one CSV file per data set, one row per value (format of the dummy data, ~/ePCP_dummy_data/README.md):
dataset_id, source_type, reference, system, property, method, T_K, p_Pa, salt_1, m_1, salt_2, m_2, m_1_final, m_2_final,
value, value_true, unit, u_value, convention, notes. Molalities in mol per kg water.

Every data point is reduced to a target quantity of the model and a residual definition; the reduction uses only
pure-water reference data (IAPWS, ice), never the model. Residuals are relative deviations of the measured quantity,
as for PCP-SAFT (Esper et al. 2023):
- osmotic coefficient: phi, relative;
- water activity, vapor pressure, freezing and boiling temperature: the model gives ln a_w at the temperature of the
  measurement; the residual is the relative deviation of the measured quantity (a_w, vapor pressure, freezing-point
  depression, boiling-point elevation), linearized in ln a_w: r = (ln a_w,calc - ln a_w,exp) / scale, with
  1/scale = d ln(measured)/d ln a_w from IAPWS-95 (vapor: pure water, fugacities) and the melting properties of ice;
- mean ionic activity coefficient (molality scale): gamma_pm, relative;
- density: density, relative (the model density is V_phi of the model + IAPWS density of pure water);
- apparent molar volume, apparent molar heat capacity, enthalpy of dilution: unchanged, scaled by the root mean
  square of the block (these quantities can be zero or change sign).
Assumptions: the vapor is pure water (not valid for volatile solutes, e.g. HCl, HNO3, NH3); the freezing point of pure
water is T_MELT (data should be given as freezing-point depression relative to the source's own pure-water value).
The data are not modified on disk; filters select rows.
"""
import glob
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import iapws  # noqa: E402  (examples/epcpsaft/iapws.py)

M_W = 0.018015268  # kg/mol, molar mass of water (IAPWS); molality <-> amount of water
R = 8.314462618
# melting of ice (freezing-point data -> water activity): J/mol, J/(mol K), K
DH_FUS, DCP_FUS, T_MELT = 6009.5, 37.9, 273.15

SALTS = {  # salt -> {ion: stoichiometric coefficient}
    "NaCl": {"sodium ion": 1, "chloride ion": 1}, "KCl": {"potassium ion": 1, "chloride ion": 1},
    "LiCl": {"lithium ion": 1, "chloride ion": 1}, "HCl": {"hydronium": 1, "chloride ion": 1},
    "NH4Cl": {"ammonium ion": 1, "chloride ion": 1}, "MgCl2": {"magnesium ion": 1, "chloride ion": 2},
    "NiCl2": {"nickel ion": 1, "chloride ion": 2}, "CoCl2": {"cobalt ion": 1, "chloride ion": 2},
    "MnCl2": {"manganese ion": 1, "chloride ion": 2},
    "LiOH": {"lithium ion": 1, "hydroxide": 1},
    "KNO3": {"potassium ion": 1, "nitrate ion": 1}, "NaNO3": {"sodium ion": 1, "nitrate ion": 1},
    "HNO3": {"hydronium": 1, "nitrate ion": 1},
    "Ni(NO3)2": {"nickel ion": 1, "nitrate ion": 2}, "Mn(NO3)2": {"manganese ion": 1, "nitrate ion": 2},
    "Li2SO4": {"lithium ion": 2, "sulfate": 1}, "Na2SO4": {"sodium ion": 2, "sulfate": 1},
    "K2SO4": {"potassium ion": 2, "sulfate": 1}, "(NH4)2SO4": {"ammonium ion": 2, "sulfate": 1},
    "H2SO4": {"hydronium": 2, "sulfate": 1},  # fully dissociated (no HSO4- speciation)
    "MgSO4": {"magnesium ion": 1, "sulfate": 1}, "NiSO4": {"nickel ion": 1, "sulfate": 1},
    "CoSO4": {"cobalt ion": 1, "sulfate": 1}, "MnSO4": {"manganese ion": 1, "sulfate": 1},
    "CuSO4": {"copper ion": 1, "sulfate": 1},
    "K2CO3": {"potassium ion": 2, "carbonate": 1}, "KHCO3": {"potassium ion": 1, "hydrogen-carbonate": 1},
    "NaOH": {"sodium ion": 1, "hydroxide": 1}, "KOH": {"potassium ion": 1, "hydroxide": 1},
    "Na2CO3": {"sodium ion": 2, "carbonate": 1}, "NaHCO3": {"sodium ion": 1, "hydrogen-carbonate": 1},
    "Li2CO3": {"lithium ion": 2, "carbonate": 1}, "Co(NO3)2": {"cobalt ion": 1, "nitrate ion": 2},
}

# property -> (target quantity, residual type); "relative": (calc - exp) / exp,
# "scaled": (calc - exp) / (root mean square of exp in the block), for quantities that can be zero or change sign,
# "linearized": (calc - exp) / scale with a scale per point (relative deviation of the measured quantity)
TARGETS = {
    "osmotic_coefficient": ("phi", "relative"),
    "water_activity": ("ln_aw", "linearized"),
    "vapor_pressure": ("ln_aw", "linearized"),
    "freezing_temperature": ("ln_aw", "linearized"),
    "boiling_temperature": ("ln_aw", "linearized"),
    "mean_ionic_activity_coefficient": ("gamma_pm", "relative"),
    "density": ("density", "relative"),
    "apparent_molar_volume": ("V_phi", "scaled"),
    "apparent_molar_heat_capacity": ("Cp_phi", "scaled"),
    "enthalpy_of_dilution": ("dil_H", "scaled"),
}


def data_path(path=None):
    path = path or os.environ.get("EPCPSAFT_ELECTROLYTE_DATA")
    if path is None:
        raise RuntimeError("set EPCPSAFT_ELECTROLYTE_DATA to the folder with the data CSV files")
    return path


def read(path=None):
    """all data rows of all CSV files in `path` (files starting with '_' are skipped)"""
    files = sorted(f for f in glob.glob(os.path.join(data_path(path), "*.csv")) if not os.path.basename(f).startswith("_"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def ions_of(composition):
    """{salt: molality} -> {ion: molality}"""
    out = {}
    for salt, m in composition.items():
        for ion, nu in SALTS[salt].items():
            out[ion] = out.get(ion, 0.0) + nu * m
    return out


def _composition(row, final=False):
    comp = {}
    for k in ("1", "2"):
        salt = row[f"salt_{k}"]
        m = row[f"m_{k}_final"] if final else row[f"m_{k}"]
        if isinstance(salt, str) and salt and np.isfinite(m):
            comp[salt] = comp.get(salt, 0.0) + float(m)
    return comp


def _ln_aw_ice(T):
    """water activity of a solution in equilibrium with ice at T (melting properties of ice)"""
    return -DH_FUS / R * (1 / T - 1 / T_MELT) - DCP_FUS / R * (1 - T_MELT / T - np.log(T / T_MELT))


def _derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def _linearized(prop, T, p, v):
    """(T, p, ln a_w,exp, scale, measured) of a colligative data point: r = (ln a_w,calc - ln a_w,exp) / scale is the
    relative deviation of the measured quantity (a_w, p, freezing-point depression or boiling-point elevation) to first
    order"""
    if prop == "water_activity":
        return T, p, np.log(v), 1.0, v                                # relative deviation of a_w
    if prop == "vapor_pressure":
        if not np.isfinite(p):
            p = v                                                     # liquid at its vapor pressure
        ln_aw = np.log(iapws.water_activity(T, v))
        dlnp = _derivative(lambda y: np.log(iapws.vapor_pressure(T, np.exp(y))), ln_aw, 1e-4)
        return T, p, ln_aw, 1.0 / dlnp, v                             # relative deviation of p
    if prop == "freezing_temperature":
        theta = T_MELT - v                                            # freezing-point depression
        slope = _derivative(_ln_aw_ice, v, 1e-3)                      # d ln a_w / dT along the ice curve
        return v, 101325.0, _ln_aw_ice(v), -theta * slope, theta      # relative deviation of theta
    if prop == "boiling_temperature":
        p = 101325.0
        theta = v - iapws.saturation_temperature(p)                   # boiling-point elevation
        slope = _derivative(lambda t: np.log(iapws.water_activity(t, p)), v, 1e-3)
        return v, p, np.log(iapws.water_activity(v, p)), theta * slope, theta   # relative deviation of theta
    raise ValueError(prop)


@dataclass
class Point:
    """one data point reduced to its target quantity"""
    index: int
    dataset_id: str
    system: str
    property: str
    target: str
    residual: str
    T: float
    p: float
    composition: dict            # {salt: molality} of the (initial) solution
    composition_final: dict = field(default_factory=dict)   # enthalpy of dilution: final solution
    value: float = np.nan        # experimental target value
    ions: tuple = ()
    scale: float = None          # residual scale of "linearized" points
    measured: float = np.nan     # measured quantity (a_w, p, freezing-point depression, boiling-point elevation or value)


def reduce(df, value_column="value", properties=None):
    """data rows -> list of Point (target quantities); rows with a property not in TARGETS or a salt not in SALTS are
    skipped (returned as second element with the reason)"""
    points, skipped = [], []
    for i, r in df.iterrows():
        prop = r["property"]
        if properties is not None and prop not in properties:
            continue
        comp = _composition(r)
        if prop not in TARGETS or not comp or any(s not in SALTS for s in comp):
            skipped.append((i, "property or salt not supported"))
            continue
        target, residual = TARGETS[prop]
        v = float(r[value_column])
        T, p = float(r["T_K"]) if np.isfinite(r["T_K"]) else np.nan, float(r["p_Pa"])
        scale, measured = None, v
        if residual == "linearized":
            T, p, v, scale, measured = _linearized(prop, T, p, v)
        comp_f = _composition(r, final=True) if prop == "enthalpy_of_dilution" else {}
        ions = tuple(sorted(ions_of(comp)))
        points.append(Point(i, r["dataset_id"], r["system"], prop, target, residual, T, p, comp, comp_f, v, ions,
                            scale, measured))
    return points, skipped
