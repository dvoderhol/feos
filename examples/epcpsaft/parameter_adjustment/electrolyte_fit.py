"""Simultaneous fit of the ion parameters of ePCP-SAFT to electrolyte data (all ions, all systems at once).

Model: ePCP-SAFT (feos `EquationOfState.epcpsaft`, variant "dh" (Debye-Hückel) or "msa" (mean spherical approximation,
packing term over the ions or over all species)) with fixed PCP-SAFT water (parameters/epcpsaft/esper2023.json).
Fitted per ion: sigma, epsilon_k, k_ij(water, ion), dipole_scaling (a11 mu^2), correlation_integral_parameter (a2;
or a2 of water for all ions, `permittivity_variant` "PT1"); global: born_delta_cation, born_delta_anion. Apparent molar
heat capacities (`Model` cp_model): residual model plus the ideal-gas heat capacities of the ions (`ideal_gas_cp`),
or an adjusted offset per salt system, composition and temperature. Not fitted (fixed): polarizability_scaling (= 0),
cation-anion k_ij (= 0; like-charged ions: 1 in feos). Bounds (BOUNDS) only where physics requires them.

Exact symmetries of the model (no data can determine them):
- permittivity gauge: a_i -> a_i + c z_i (a = dipole_scaling or correlation_integral_parameter) leaves every
  electroneutral solution unchanged;
- dispersion symmetry: ions interact by dispersion only with water and with oppositely charged ions (k_ij = 1 for
  like-charged ions, including i = j), and only eps_ij = sqrt(eps_i eps_j)(1 - k_ij) enters the model. Therefore
  sqrt(eps_i) -> lambda sqrt(eps_i), (1 - k_iw) -> (1 - k_iw) / lambda for all cations and sqrt(eps_i) -> sqrt(eps_i) / lambda,
  (1 - k_iw) -> lambda (1 - k_iw) for all anions leaves every eps_ij unchanged (`dispersion_symmetry`).
During the fit the parameters GAUGE_PARAMETERS (permittivity parameters and eps_k) of one reference ion per group of
connected ions are fixed (`gauge_fixed`; Cl- where present). eps_k of the reference ion is set to its literature value
(REFERENCE_EPSILON_K: Cl- 170 K, Held et al. 2014) by the dispersion symmetry (`reference_gauge`), so that all other
ion eps_k are on the familiar ePC-SAFT scale; predictions are unchanged by this. After the fit, `macinnes_gauge` shifts
the permittivity parameters so that gamma(K+) = gamma(Cl-) in aqueous KCl (MacInnes convention).

Objective (as for PCP-SAFT, Esper et al. 2023, eqs. 2-5): sum over blocks b (system, property) of
w_b / N_b * sum_i L(r_i), L = Huber loss with delta (r^2/2 for |r| <= delta, delta (|r| - delta/2) otherwise),
r relative (or block-scaled, electrolyte_data.TARGETS). The residual vector passed to scipy.optimize.least_squares is
f_i = sqrt(w_b / N_b) psi(r_i), psi(r) = r for |r| <= delta, sign(r) sqrt(2 delta |r| - delta^2) otherwise, so that
0.5 * sum f_i^2 equals the objective; the Jacobian is exact (feos automatic differentiation).

Optimization (`Problem`, `multistart`, `continuation`, `robust_fit`):
1. problem setup that removes local minima and long curved valleys before an optimizer sees them: exact symmetries
   removed (above); optimizer variables s_i = sqrt(eps_i) and u_i = sqrt(eps_i) (1 - k_iw) per ion instead of
   (eps_i, k_iw) (`cross_energies=True`): the ion-water dispersion energy depends only on u_i, cation-anion energies
   only on s_i, which removes the strong eps_i-k_iw correlation; continuation from easy to all data
   (`continuation`: e.g. first low molalities near 298 K, then all data);
2. gradient-based nonlinear least squares with robust (Huber) loss: scipy.optimize.least_squares (trust region
   reflective, bounds, exact Jacobian);
3. many starting points: scrambled Sobol sample of the parameter box plus given starts; every start is screened with
   a limited number of evaluations, the best ones are converged; `classify` decides from the converged results whether
   the problem is unimodal, has flat valleys (equal cost, different parameters: not identifiable) or is multimodal;
4. fallback for true multimodality: CMA-ES (`cma_es`, pycma) on the same objective, polished by least squares.
"""
import json
import os
from dataclasses import dataclass, field

import feos
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

import electrolyte_data as ed

M_W = ed.M_W
ION_PARAMETERS = ("sigma", "epsilon_k", "k_w", "dipole_scaling", "correlation_integral_parameter")
GLOBAL_PARAMETERS = ("born_delta_cation", "born_delta_anion")
# optimizer bounds and multistart sampling box per parameter type
# bounds only where physics requires them: sigma > 0, eps_k >= 0, ion-water dispersion energy
# sqrt(eps_i eps_w) (1 - k_iw) >= 0, i.e. k_iw <= 1
BOUNDS = {"sigma": (0.0, np.inf), "epsilon_k": (0.0, np.inf), "k_w": (-np.inf, 1.0),
          "dipole_scaling": (-np.inf, np.inf), "correlation_integral_parameter": (-np.inf, np.inf),
          "born_delta_cation": (-np.inf, np.inf), "born_delta_anion": (-np.inf, np.inf), "Cp_offset": (-np.inf, np.inf),
          "ion_pairing": (-np.inf, np.inf)}
SAMPLING = {"sigma": (2.0, 4.5), "epsilon_k": (50.0, 2000.0), "k_w": (-0.5, 0.5), "dipole_scaling": (0.0, 12.0),
            "correlation_integral_parameter": (0.0, 0.5), "born_delta_cation": (-0.3, 0.3),
            "born_delta_anion": (-0.3, 0.3), "Cp_offset": (0.0, 0.0), "ion_pairing": (-2.0, 2.0)}
# exact symmetries (see module docstring): parameters of a reference ion fixed per group of connected ions
GAUGE_PARAMETERS = ("dipole_scaling", "correlation_integral_parameter", "epsilon_k")
REFERENCE_ION = "chloride ion"
# eps_k of reference ions (dispersion symmetry): literature values
REFERENCE_EPSILON_K = {"chloride ion": 170.0}     # Held et al. (2014)
# typical magnitudes (scaling of the Jacobian in the identifiability analysis)
TYPICAL = {"sigma": 1.0, "epsilon_k": 100.0, "k_w": 0.1, "dipole_scaling": 1.0, "correlation_integral_parameter": 0.1,
           "born_delta_cation": 0.1, "born_delta_anion": 0.1, "Cp_offset": 10.0, "ion_pairing": 1.0}
METRICS = ("MAD", "MARD%", "RMSE", "RMSRE%", "bias", "rel_bias%", "max_AD", "max_RD%")


# ---------------------------------------------------------------- parameters (feos JSON format)
def read_records(pure_file, binary_file=None):
    """pure records (dicts) by name and water-ion k_ij by ion name from feos JSON files"""
    pure = {r["identifier"]["name"]: r for r in json.load(open(pure_file))}
    k_w = {}
    if binary_file:
        for b in json.load(open(binary_file)):
            names = {b["id1"]["name"], b["id2"]["name"]}
            if "water" in names and len(names) == 2:
                k_w[(names - {"water"}).pop()] = b.get("k_ij", 0.0)
    return pure, k_w


def water_record(file=None):
    """PCP-SAFT water with permittivity record (parameters/epcpsaft/esper2023.json of this repository)"""
    file = file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "parameters",
                                "epcpsaft", "esper2023.json")
    return next(r for r in json.load(open(file)) if r["identifier"].get("cas") == "7732-18-5")


@dataclass
class ParameterSet:
    """Vector of all adjustable parameters (ion parameters, Born corrections, Cp offsets) and their mapping."""
    ions: list                       # ion names
    records: dict                    # ion name -> pure record (dict): molarweight, z, ...
    names: list                      # (kind, ion or None, parameter type) for every entry of x
    x0: np.ndarray                   # start / current values
    free: np.ndarray                 # mask of adjusted entries
    water: dict = field(default_factory=dict)

    @classmethod
    def new(cls, ions, pure, k_w, water, born_delta=(0.0, 0.0), cp_blocks=(), fixed=(), pairs=()):
        """`fixed`: iterable of (ion, parameter type) that are not adjusted (e.g. the gauge of the reference ion);
        `pairs`: (cation, anion) pairs with the ion-pairing parameter (logarithm of the factor of the Bjerrum
        association constant, start value 0; key "cation|anion"; variants "bidh" and "bimsa")"""
        names, x = [], []
        for ion in ions:
            r = pure[ion]
            pt = r["permittivity_record"]["PerturbationTheory"]
            vals = {"sigma": r["sigma"], "epsilon_k": r["epsilon_k"], "k_w": k_w.get(ion, 0.0),
                    "dipole_scaling": pt["dipole_scaling"], "correlation_integral_parameter": pt["correlation_integral_parameter"]}
            for p in ION_PARAMETERS:
                names.append(("ion", ion, p)); x.append(vals[p])
        for p, v in zip(GLOBAL_PARAMETERS, born_delta):
            names.append(("global", None, p)); x.append(v)
        for b in cp_blocks:
            names.append(("Cp_offset", b, "Cp_offset")); x.append(0.0)
        for c, a in pairs:
            names.append(("pair", f"{c}|{a}", "ion_pairing")); x.append(0.0)
        free = np.array([(n[1], n[2]) not in set(fixed) for n in names])
        return cls(list(ions), {i: pure[i] for i in ions}, names, np.array(x, float), free, water)

    def index(self, kind, key, parameter):
        return self.names.index((kind, key, parameter))

    def bounds(self):
        lo = np.array([BOUNDS[n[2]][0] for n in self.names])
        hi = np.array([BOUNDS[n[2]][1] for n in self.names])
        return lo[self.free], hi[self.free]

    def sampling_box(self):
        lo = np.array([SAMPLING[n[2]][0] for n in self.names])
        hi = np.array([SAMPLING[n[2]][1] for n in self.names])
        return lo[self.free], hi[self.free]

    def at(self, x):
        """copy with the full vector x as values of all (also the fixed) parameters"""
        return ParameterSet(self.ions, self.records, self.names, np.array(x, float), self.free.copy(), self.water)

    def full(self, x_free):
        x = self.x0.copy()
        x[self.free] = x_free
        return x

    def labels(self, free_only=True):
        lab = [f"{n[2]}[{n[1]}]" if n[1] is not None else n[2] for n in self.names]
        return [l for l, f in zip(lab, self.free) if f] if free_only else lab

    def ion_records(self, x):
        """(pure records, water-ion binary records) in feos JSON format for the full vector x"""
        pure, binary = [], []
        for ion in self.ions:
            v = {p: x[self.index("ion", ion, p)] for p in ION_PARAMETERS}
            r = {k: val for k, val in self.records[ion].items() if k != "source"}
            r = dict(r, sigma=float(v["sigma"]), epsilon_k=float(v["epsilon_k"]), permittivity_record={"PerturbationTheory": {
                "dipole_scaling": float(v["dipole_scaling"]), "polarizability_scaling": 0.0,
                "correlation_integral_parameter": float(v["correlation_integral_parameter"])}})
            pure.append(r)
            binary.append({"id1": {"name": "water"}, "id2": {"name": ion}, "k_ij": float(v["k_w"])})
        for i, n in enumerate(self.names):
            if n[0] == "pair":
                c, a = n[1].split("|")
                binary.append({"id1": {"name": c}, "id2": {"name": a}, "k_ij": 0.0, "ion_pairing": float(x[i])})
        return pure, binary

    def pairs(self):
        """(cation, anion, index in x) of the ion-pairing parameters"""
        return [(*n[1].split("|"), i) for i, n in enumerate(self.names) if n[0] == "pair"]

    def born_delta(self, x):
        return [x[self.index("global", None, p)] for p in GLOBAL_PARAMETERS]


# ---------------------------------------------------------------- model
class Model:
    """Target quantities of all points and their derivatives with respect to the parameter vector."""

    def __init__(self, parameters, points, variant="dh", msa_packing="ions", cp_model="ideal_gas",
                 cp_electronic=False, reactions=(), ion_pairing_constant="bjerrum"):
        """variant: term for the ion-ion interactions, "dh" (Debye-Hückel), "msa" (mean spherical approximation),
          "bidh" (Debye-Hückel with ion pairing) or "bimsa" (binding MSA);
        msa_packing: species in the packing term of the MSA, "ions" (primitive model) or "total" (all species);
        cp_model: apparent molar heat capacity = residual part of the model plus
          "ideal_gas": the ideal-gas heat capacities of the ions (ideal_gas_cp), no adjustable parameter;
          "offset": one adjusted offset per Cp block (parameters "Cp_offset" of the parameter set, cp_block);
          "residual": nothing (residual convention);
        cp_electronic: with cp_model "ideal_gas", include the electronic heat capacity of the ions (electronic_cp);
        reactions: reaction equilibria (e.g. REACTIONS) of the systems that contain all reactants or all products of a
          reaction (system_species); the compositions of the data are stoichiometric, every reactant that is the
          only reactant of a reaction is replaced by its products in the amounts passed to feos (basis_moles);
        ion_pairing_constant: association constant of the cation-anion pairs of "bidh" and "bimsa", "bjerrum" or
          "ebeling"."""
        if cp_model not in ("ideal_gas", "offset", "residual"):
            raise ValueError(f"cp_model {cp_model!r}")
        self.par = parameters
        self.points = points
        self.variant = variant
        self.msa_packing = msa_packing
        self.ion_pairing_constant = ion_pairing_constant
        self.cp_model = cp_model
        self.cp_electronic = cp_electronic
        self.reactions = tuple(reactions)
        self.systems = {}
        for k, pt in enumerate(points):
            self.systems.setdefault(system_species(pt.ions, self.reactions), []).append(k)
        missing = {i for ions in self.systems for i in ions} - set(parameters.ions)
        if missing:
            raise ValueError(f"no parameters for ions {sorted(missing)} (model_ions gives all species)")
        # IAPWS density of pure water at the states of density points (independent of the parameters)
        self.rho_w = {(p.T, p.p): iapws.rho_w(p.T, p.p) for p in points if p.target == "density"}

    def _eos_parameters(self, ions, x):
        pure, binary = self.par.ion_records(x)
        pure = {r["identifier"]["name"]: r for r in pure}
        recs = [feos.PureRecord.from_json_str(json.dumps({k: v for k, v in self.par.water.items() if k != "source"}))]
        recs += [feos.PureRecord.from_json_str(json.dumps(pure[i])) for i in ions]
        brs = [feos.BinaryRecord.from_json_str(json.dumps(b)) for b in binary
               if b["id2"]["name"] in ions and (b["id1"]["name"] == "water" or b["id1"]["name"] in ions)]
        return feos.Parameters.from_records(recs, brs, feos.IdentifierOption.Name)

    def _derivatives(self, ions):
        """(feos derivative specification, index in the full vector) of all parameters that act on `ions`"""
        spec, cols = [], []
        for c, ion in enumerate(ions, start=1):
            for p in ION_PARAMETERS:
                name, idx = ("k_ij", [0, c]) if p == "k_w" else (p, [c])
                spec.append((name, idx)); cols.append(self.par.index("ion", ion, p))
        for p in GLOBAL_PARAMETERS:
            spec.append((p, [])); cols.append(self.par.index("global", None, p))
        if self.variant in ("bidh", "bimsa"):
            for c, a, k in self.par.pairs():
                if c in ions and a in ions:
                    spec.append(("ion_pairing", [ions.index(c) + 1, ions.index(a) + 1])); cols.append(k)
        return spec, cols

    @staticmethod
    def _moles(ions, composition):
        mi = ed.ions_of(composition)
        return [1.0 / M_W] + [mi.get(i, 0.0) for i in ions]

    def evaluate(self, x, jacobian=True):
        """target values (n) and derivatives (n x len(x)) for the full parameter vector x; NaN where not converged"""
        n, nx = len(self.points), len(x)
        values = np.full(n, np.nan)
        jac = np.zeros((n, nx)) if jacobian else None
        d_cat, d_an = self.par.born_delta(x)
        for ions, idx in self.systems.items():
            params = self._eos_parameters(ions, x)
            spec, cols = self._derivatives(ions) if jacobian else ([], [])
            rows, owner = [], []
            for k in idx:
                pt = self.points[k]
                rows.append([pt.T, pt.p] + basis_moles(ions, pt.composition, self.reactions)); owner.append((k, 0))
                if pt.target == "dil_H":
                    rows.append([pt.T, pt.p] + basis_moles(ions, pt.composition_final, self.reactions))
                    owner.append((k, 1))
            v, g, ok = feos.Property.electrolyte_properties_derivatives(
                params, spec, np.array(rows), solvent=0, epcpsaft_variant=self.variant,
                msa_packing=self.msa_packing, born_delta_cation=d_cat, born_delta_anion=d_an,
                reactions=feos_reactions(ions, self.reactions), solvent_molar_mass=M_W,
                ion_pairing_constant=self.ion_pairing_constant)
            state = {o: (_row(v, j), _row(g, j) if jacobian else None, ok[j]) for j, o in enumerate(owner)}
            for k in idx:
                pt = self.points[k]
                val, grad = self._target(pt, ions, state[(k, 0)], state.get((k, 1)), x)
                if val is None:
                    continue
                values[k] = val
                if jacobian:
                    jac[k, cols] = grad[0]
                    if grad[1] is not None:
                        jac[k, grad[1][0]] += grad[1][1]
        return values, jac

    def _target(self, pt, ions, s0, s1, x):
        """target value and (derivative w.r.t. the system parameters, (index, derivative) of a Cp offset)"""
        v, g, ok = s0
        if not ok or (s1 is not None and not s1[2]):
            return None, None
        mi = ed.ions_of(pt.composition)
        nu_m = sum(mi.values())
        m_salt = sum(pt.composition.values())
        basis = basis_moles(ions, pt.composition, self.reactions)
        gz = (lambda key, *i: g[key][i] if i else g[key]) if g is not None else (lambda key, *i: 0.0)
        extra = None
        if pt.target == "phi":
            val, d = -v["ln_solvent_activity"] / (M_W * nu_m), -gz("ln_solvent_activity") / (M_W * nu_m)
        elif pt.target == "ln_aw":
            val, d = v["ln_solvent_activity"], gz("ln_solvent_activity")
        elif pt.target == "gamma_pm":
            # stoichiometric mean ionic activity coefficient (molality scale) of the electrolyte:
            # gamma_pm^nu = prod_i (m_i gamma_i^m / m_i^st)^nu_i over its ions, m_i (true) and gamma_i^m of the
            # (equilibrium) solution, m_i^st the stoichiometric molality of ion i; amounts per kg water: m_i = n_i
            salt = pt.electrolyte or next(iter(pt.composition))
            nu = ed.SALTS[salt]
            n = v["moles"]
            x_w = n[0] / n.sum()
            dn = g["moles"] if g is not None else 0.0
            d_ln_xw = -(dn.sum(axis=0)) / n.sum() if g is not None else 0.0
            c = {i: 1 + ions.index(i) for i in nu}
            ln_g = sum(k * (np.log(n[c[i]] / mi[i]) + v["ln_gamma"][c[i]] + np.log(x_w))
                       for i, k in nu.items()) / sum(nu.values())
            d_ln = sum(k * ((dn[c[i]] / n[c[i]] if g is not None else 0.0) + gz("ln_gamma", c[i]) + d_ln_xw)
                       for i, k in nu.items()) / sum(nu.values())
            val, d = np.exp(ln_g), np.exp(ln_g) * d_ln
        elif pt.target == "density":
            mass = 1.0 + sum(self.par.records[i]["molarweight"] / 1000 * mi[i] for i in mi)
            vol = 1.0 / self.rho_w[(pt.T, pt.p)] + v["volume_difference"]
            val, d = mass / vol, -mass / vol ** 2 * gz("volume_difference")
        elif pt.target == "V_phi":
            val, d = v["volume_difference"] / m_salt, gz("volume_difference") / m_salt
        elif pt.target == "Cp_phi":
            key = "residual_heat_capacity_difference"
            val, d = v[key] / m_salt, gz(key) / m_salt
            if self.cp_model == "offset":
                j = self.par.index("Cp_offset", cp_block(pt), "Cp_offset")
                val, extra = val + x[j], (j, 1.0)
            elif self.cp_model == "ideal_gas":
                val += sum(nb * ideal_gas_cp(i, pt.T, self.cp_electronic)
                           for i, nb in zip(ions, basis[1:]) if nb > 0) / m_salt
        elif pt.target == "dil_H":
            vf, g_final, _ = s1
            key = "relative_enthalpy"
            ratio = m_salt / sum(pt.composition_final.values())
            val = (ratio * vf[key] - v[key]) / m_salt
            d = (ratio * (g_final[key] if g_final is not None else 0.0) - gz(key)) / m_salt
        else:
            raise ValueError(pt.target)
        return val, (d, extra)


# reaction equilibria (molality scale, ln K(T) of feos ReactionEquilibrium, Knopf et al. 2003, eq. 16): nu over species,
# ln K at t0, Delta H / R (K), Delta c_p / R, (d Delta c_p / dT) / R (1/K) at t0
REACTIONS = (
    dict(name="HSO4- = H+ + SO4 2-", nu={"hydrogen-sulfate": -1, "hydronium": 1, "sulfate": 1},
         ln_k0=float(np.log(1.0576e-2)), t0=298.15, dh0=-2231.620793, dcp0=-24.7273, dcp_dt=-0.11967,
         source="Knopf, Luo, Krieger, Koop, J. Phys. Chem. A 107 (2003) 4322, Table 2"),
)


def _relevant(reaction, species):
    """a reaction takes part if all its reactants or all its products are present"""
    s = set(species)
    reactants = {i for i, n in reaction["nu"].items() if n < 0}
    products = {i for i, n in reaction["nu"].items() if n > 0}
    return reactants <= s or products <= s


def system_species(ions, reactions=()):
    """species of a system: its ions and all species of the reactions that take part"""
    species = set(ions)
    changed = True
    while changed:
        changed = False
        for r in reactions:
            if _relevant(r, species) and not set(r["nu"]) <= species:
                species |= set(r["nu"])
                changed = True
    return tuple(sorted(species))


def model_ions(points, reactions=()):
    """all species of the systems of `points` (ions of the parameter set)"""
    return sorted({i for p in points for i in system_species(p.ions, reactions)})


def basis_moles(species, composition, reactions=()):
    """amounts (water 1/M_w, then `species`) of the stoichiometric `composition`, with every species that is the
    only reactant of a reaction taking part replaced by the products of that reaction"""
    mi = dict(ed.ions_of(composition))
    for r in reactions:
        if not _relevant(r, species):
            continue
        reactants = [i for i, n in r["nu"].items() if n < 0]
        if len(reactants) == 1 and mi.get(reactants[0], 0.0) > 0:
            a = mi.pop(reactants[0]) / -r["nu"][reactants[0]]
            for i, n in r["nu"].items():
                if n > 0:
                    mi[i] = mi.get(i, 0.0) + n * a
    return [1.0 / M_W] + [mi.get(i, 0.0) for i in species]


def feos_reactions(species, reactions=()):
    """reactions taking part in the system `species` in the format of feos (stoichiometry over water and species)"""
    return [([0.0] + [float(r["nu"].get(i, 0)) for i in species], r["ln_k0"], r["t0"], r["dh0"], r["dcp0"],
             r["dcp_dt"]) for r in reactions if _relevant(r, species)]


def _row(results, j):
    """row j of the dict of result arrays of feos.Property.electrolyte_properties_derivatives"""
    return {k: a[j] for k, a in results.items()}


R = 8.314462618
# ideal-gas heat capacity of ions (gas-phase ions as reference state of the residual model):
# c_p = 5/2 R (translation and pV) + rotation (R linear, 3/2 R nonlinear) + harmonic vibrations (Einstein functions);
# electronic contributions are not included. Vibrational wavenumbers in cm^-1 with degeneracy: fundamentals of the ions
# in aqueous solution (Nakamoto, Infrared and Raman Spectra of Inorganic and Coordination Compounds, Part A, 6th ed.,
# Wiley 2009).
MONATOMIC_IONS = {"hydronium", "lithium ion", "sodium ion", "potassium ion", "rubidium ion", "cesium ion",
                  "silver ion", "thallium ion", "magnesium ion", "calcium ion", "strontium ion", "barium ion",
                  "manganese ion", "iron(2+) ion", "cobalt ion", "nickel ion", "copper ion", "zinc ion",
                  "cadmium ion", "lead ion", "beryllium ion", "aluminium ion", "fluoride ion", "chloride ion",
                  "bromide ion", "iodide ion"}
POLYATOMIC_IONS = {   # ion: (linear, [(wavenumber / cm^-1, degeneracy), ...])
    "hydroxide": (True, [(3555.0, 1)]),
    "ammonium ion": (False, [(3040.0, 1), (1680.0, 2), (3145.0, 3), (1400.0, 3)]),
    "nitrate ion": (False, [(1050.0, 1), (831.0, 1), (1390.0, 2), (720.0, 2)]),
    "sulfate": (False, [(983.0, 1), (450.0, 2), (1105.0, 3), (611.0, 3)]),
    "perchlorate ion": (False, [(935.0, 1), (460.0, 2), (1100.0, 3), (630.0, 3)]),
    "carbonate": (False, [(1063.0, 1), (879.0, 1), (1415.0, 2), (680.0, 2)]),
    "chlorate ion": (False, [(930.0, 1), (610.0, 1), (975.0, 2), (480.0, 2)]),
    "bromate ion": (False, [(805.0, 1), (418.0, 1), (805.0, 2), (358.0, 2)]),
    "thiocyanate ion": (True, [(2066.0, 1), (743.0, 1), (486.0, 2)]),
}
C2_CM_K = 1.438776877  # second radiation constant h c / k_B in cm K
# electronic levels of the free ions (degeneracy 2J+1, energy / cm^-1): all levels below 10000 cm^-1 (the next levels
# lie above 14000 cm^-1); NIST Atomic Spectra Database, retrieved 2026-11-09 (Mn III, Ni III: Sugar and Corliss 1985;
# Fe III: Ekberg 1993, Sugar and Corliss 1985; Co III: Smillie et al. 2016; Cu III: Sugar and Musgrove 1990). Ions not
# listed have no level below 10000 cm^-1 that contributes (closed shells, Mn2+ 3d5 6S).
ELECTRONIC_LEVELS = {
    "manganese ion": [(6, 0.0)],
    "iron(2+) ion": [(9, 0.0), (7, 436.19), (5, 738.87), (3, 932.52), (1, 1027.26)],
    "cobalt ion": [(10, 0.0), (8, 841.36), (6, 1451.33), (4, 1867.50)],
    "nickel ion": [(9, 0.0), (7, 1360.7), (5, 2269.6)],
    "copper ion": [(6, 0.0), (4, 2071.69)],
}


def electronic_cp(ion, T):
    """electronic heat capacity of a free ion [J/(mol K)]: R (<e^2> - <e>^2), e = E / (k T), Boltzmann distribution
    over ELECTRONIC_LEVELS (0 for ions without listed levels)"""
    levels = ELECTRONIC_LEVELS.get(ion, [(1, 0.0)])
    e = np.array([C2_CM_K * E / T for _, E in levels])
    w = np.array([g for g, _ in levels]) * np.exp(-e)
    w /= w.sum()
    return R * (w @ e ** 2 - (w @ e) ** 2)


def ideal_gas_cp(ion, T, electronic=False):
    """ideal-gas isobaric heat capacity of an ion [J/(mol K)] (MONATOMIC_IONS, POLYATOMIC_IONS); with `electronic`,
    the electronic contribution of electronic_cp is added"""
    el = electronic_cp(ion, T) if electronic else 0.0
    if ion in MONATOMIC_IONS:
        return 2.5 * R + el
    if ion not in POLYATOMIC_IONS:
        raise ValueError(f"no ideal-gas heat capacity for {ion!r}")
    linear, modes = POLYATOMIC_IONS[ion]
    vib = 0.0
    for nu, g in modes:
        u = C2_CM_K * nu / T
        vib += g * u ** 2 * np.exp(u) / np.expm1(u) ** 2
    return R * (2.5 + (1.0 if linear else 1.5) + vib) + el


def cp_block(pt):
    """nuisance block of the apparent molar heat capacity (sum of the ideal-gas heat capacities of the ions per mol of
    salt): salt system, composition of the salt mixture and temperature"""
    m = sum(pt.composition.values())
    y = ",".join(f"{s}:{v / m:.4f}" for s, v in sorted(pt.composition.items())) if len(pt.composition) > 1 else ""
    return f"{pt.system}{'[' + y + ']' if y else ''}@{pt.T:.2f}"


def cp_isotherm(pt):
    """isotherm and isobar of the apparent molar heat capacity (cp_block and pressure): group of the removed offset
    (Objective.remove_cp_offset)"""
    return f"{cp_block(pt)}@{pt.p / 1e5:.1f}bar"


iapws = ed.iapws


# ---------------------------------------------------------------- objective
def huber(r, delta):
    """residual transformation psi(r) and its derivative: psi^2 / 2 is the Huber loss L(r) with threshold delta
    (Esper et al. 2023, eq. 3): r^2 / 2 for |r| <= delta, delta (|r| - delta / 2) otherwise"""
    a = np.abs(r)
    psi = np.where(a <= delta, r, np.sign(r) * np.sqrt(np.maximum(2 * delta * a - delta * delta, 0.0)))
    dpsi = np.where(a <= delta, 1.0, delta / np.sqrt(np.maximum(2 * delta * a - delta * delta, 1e-300)))
    return psi, dpsi


@dataclass
class Objective:
    """Huber objective with weights w_b / N_b per block (system, property), as in Esper et al. (2023), or (with
    block_normalization=False) w_b for every point."""
    model: Model
    delta: float = 0.05
    weights: dict = field(default_factory=dict)    # property -> w (default 1)
    # True: w_b / N_b per block (Esper et al. 2023); False: weight w_b for every point (blocks with more points weigh more)
    block_normalization: bool = True
    # r of a point that did not converge; None: NaN, so that least_squares rejects the step (a fixed value would make
    # non-convergence attractive for points with |r| above it)
    failure_residual: float = None
    # True: apparent molar heat capacities enter as deviations from their mean per isotherm and isobar (cp_isotherm),
    # i.e. calc - exp minus its mean over the group: the infinite-dilution (standard-state) value Cp_phi^0(T, p) of
    # every isotherm is removed and only the concentration dependence Cp_phi - Cp_phi^0 is fitted; groups with one point
    # carry no information (residual 0)
    remove_cp_offset: bool = False

    def __post_init__(self):
        pts = self.model.points
        self.exp = np.array([p.value for p in pts])
        blocks = [(p.system, p.property) for p in pts]
        self.block_names = sorted(set(blocks))
        self.block = np.array([self.block_names.index(b) for b in blocks])
        n_b = np.bincount(self.block)
        w = np.array([self.weights.get(b[1], 1.0) for b in self.block_names])
        self.sqrt_w = np.sqrt(w[self.block] / (n_b[self.block] if self.block_normalization else 1.0))
        scale = np.ones(len(pts))
        for b in range(len(self.block_names)):
            sel = self.block == b
            kind = pts[np.flatnonzero(sel)[0]].residual
            if kind == "scaled":
                scale[sel] = np.sqrt(np.mean(self.exp[sel] ** 2))
            elif kind == "linearized":
                scale[sel] = [pts[k].scale for k in np.flatnonzero(sel)]
            else:
                scale[sel] = np.abs(self.exp[sel])
        zero = sorted({self.block_names[b] for b in self.block[~(np.abs(scale) > 0)]})
        if zero:
            raise ValueError(f"residual scale zero (all values of a block zero, or zero depression/elevation): {zero}")
        self.scale = scale
        groups = [cp_isotherm(p) if (self.remove_cp_offset and p.target == "Cp_phi") else None for p in pts]
        names = sorted({g for g in groups if g is not None})
        self.cp_group = np.array([names.index(g) if g is not None else -1 for g in groups])
        self.cp_group_size = np.bincount(self.cp_group[self.cp_group >= 0], minlength=len(names))
        self._values = (None, None)
        self._jacobian = (None, None)

    def _evaluate(self, x_full, jacobian):
        """values (cached) and, if requested, Jacobian (cached) at x_full"""
        key = x_full.tobytes()
        if jacobian and self._jacobian[0] != key:
            calc, jac = self.model.evaluate(x_full, jacobian=True)
            self._jacobian, self._values = (key, jac), (key, calc)
        elif self._values[0] != key:
            self._values = (key, self.model.evaluate(x_full, jacobian=False)[0])
        return self._values[1], (self._jacobian[1] if jacobian else None)

    def relative_residuals(self, x_full):
        calc, _ = self.model.evaluate(x_full, jacobian=False)
        return (calc - self.exp) / self.scale, calc

    def _psi(self, r):
        return huber(r, self.delta)

    def _without_offset(self, d):
        """d (n or n x k, deviations or their derivatives) with the mean over every Cp group removed (finite rows only)"""
        if not np.any(self.cp_group >= 0):
            return d
        d = np.array(d, float)
        for g in np.unique(self.cp_group[self.cp_group >= 0]):
            rows = np.flatnonzero(self.cp_group == g)
            ok = rows[np.isfinite(d[rows]).reshape(len(rows), -1).all(axis=1)]
            if len(ok):
                d[rows] = d[rows] - np.mean(d[ok], axis=0)
        return d

    def deviations(self, calc):
        """calc - exp, for apparent molar heat capacities with the offset per group removed (remove_cp_offset)"""
        return self._without_offset(calc - self.exp)

    def _residual(self, calc):
        r = self.deviations(calc) / self.scale
        failed = ~np.isfinite(r)
        r[failed] = np.nan if self.failure_residual is None else self.failure_residual
        return r, failed

    def residuals(self, x_free, par):
        calc, _ = self._evaluate(par.full(x_free), jacobian=False)
        r, _ = self._residual(calc)
        psi, _ = self._psi(r)
        return self.sqrt_w * psi

    def jacobian(self, x_free, par):
        calc, jac = self._evaluate(par.full(x_free), jacobian=True)
        r, failed = self._residual(calc)
        _, dpsi = self._psi(np.where(failed, 0.0, r))
        jac = jac[:, par.free]
        if np.any(self.cp_group >= 0):
            jac = np.where(failed[:, None], np.nan, jac)
            jac = np.nan_to_num(self._without_offset(jac), nan=0.0)
        J = (self.sqrt_w * dpsi / self.scale)[:, None] * jac
        J[failed] = 0.0
        return J

    def cost(self, x_free, par):
        f = self.residuals(x_free, par)
        return 0.5 * f @ f


# ---------------------------------------------------------------- optimization
S_MIN = 1e-6   # lower bound of s = sqrt(epsilon_k / K) in the optimizer variables (epsilon_k >= 1e-12 K)


class Problem:
    """Objective in the variables of the optimizer.

    With `cross_energies=True`, (epsilon_k, k_w) of every ion are replaced by s = sqrt(eps_k) and
    u = sqrt(eps_k) (1 - k_w): the ion-water dispersion energy sqrt(eps_i eps_w) (1 - k_iw) = sqrt(eps_w) u depends on u
    only and the cation-anion energies sqrt(eps_c eps_a) = s_c s_a on s only. All other variables are the free
    parameters themselves. Results are always returned as parameters (`x_free`, `x_full`)."""

    def __init__(self, objective, par, cross_energies=True):
        self.objective, self.par, self.cross_energies = objective, par, cross_energies
        free = np.flatnonzero(par.free)
        pos = {j: k for k, j in enumerate(free)}            # full index -> free index
        self.pairs = []                                      # (free index of eps or None, free index of k_w, eps if fixed)
        if cross_energies:
            for ion in par.ions:
                je, jk = par.index("ion", ion, "epsilon_k"), par.index("ion", ion, "k_w")
                if jk in pos:
                    self.pairs.append((pos.get(je), pos[jk], None if je in pos else par.x0[je]))

    # variables <-> parameters
    def to_y(self, x_free):
        y = np.array(x_free, float)
        for ie, ik, eps in self.pairs:
            sq = np.sqrt(x_free[ie] if ie is not None else eps)
            if ie is not None:
                y[ie] = sq
            y[ik] = sq * (1.0 - x_free[ik])
        return y

    def x_free(self, y):
        x = np.array(y, float)
        for ie, ik, eps in self.pairs:
            sq = y[ie] if ie is not None else np.sqrt(eps)
            if ie is not None:
                x[ie] = sq ** 2
            x[ik] = 1.0 - y[ik] / sq
        return x

    def x_full(self, y):
        return self.par.full(self.x_free(y))

    def _dx_dy(self, y, J):
        """J (d residuals / d x_free) -> d residuals / d y"""
        J = J.copy()
        for ie, ik, eps in self.pairs:
            sq = y[ie] if ie is not None else np.sqrt(eps)
            col_k = J[:, ik].copy()
            J[:, ik] = col_k * (-1.0 / sq)
            if ie is not None:
                J[:, ie] = J[:, ie] * 2.0 * sq + col_k * y[ik] / sq ** 2
        return J

    def residuals(self, y):
        return self.objective.residuals(self.x_free(y), self.par)

    def jacobian(self, y):
        return self._dx_dy(y, self.objective.jacobian(self.x_free(y), self.par))

    def cost(self, y):
        f = self.residuals(y)
        return 0.5 * f @ f

    def bounds(self):
        lo, hi = self.par.bounds()
        lo, hi = lo.copy(), hi.copy()
        for ie, ik, eps in self.pairs:
            k_lo, k_hi = lo[ik], hi[ik]
            if ie is not None:
                # s >= S_MIN: at s = 0 the chain rule of the variables (division by s) is undefined
                s_lo, s_hi = max(np.sqrt(lo[ie]), S_MIN), np.sqrt(hi[ie])
                lo[ie], hi[ie] = s_lo, s_hi
            else:
                s_lo = s_hi = np.sqrt(eps)
            lo[ik], hi[ik] = s_lo * (1.0 - k_hi), s_hi * (1.0 - k_lo)
        return lo, hi

    def sample(self, n, seed=0):
        """n starting points: scrambled Sobol sample of the parameter sampling box (SAMPLING), as variables y"""
        lo, hi = self.par.sampling_box()
        m = int(np.ceil(np.log2(max(n, 1))))
        u = qmc.Sobol(d=len(lo), scramble=True, seed=seed).random_base2(m)[:n]   # first n of the next power of 2
        return [self.to_y(x) for x in qmc.scale(u, lo, np.where(hi > lo, hi, lo + 1e-12))]


def fit(problem, y0, max_nfev=None, verbose=0, **kwargs):
    """one local optimization: trust region reflective with bounds and exact Jacobian (as for PCP-SAFT)"""
    lo, hi = problem.bounds()
    y0 = np.clip(y0, lo, hi)
    return least_squares(problem.residuals, y0, jac=problem.jacobian, bounds=(lo, hi), method="trf", x_scale="jac",
                         max_nfev=max_nfev, verbose=verbose, **kwargs)


def multistart(problem, n_starts=20, starts=(), seed=0, screening_nfev=15, n_polish=3, polish_nfev=None,
               verbose=False, **kwargs):
    """Multistart: the given `starts` (variables y) plus a Sobol sample of n_starts points; every start runs for
    `screening_nfev` evaluations, the `n_polish` best are converged. Returns all results sorted by cost (dicts with
    y, x (free parameters), cost, start, polished, status, optimality)."""
    candidates = [np.asarray(y, float) for y in starts] + (problem.sample(n_starts, seed) if n_starts else [])
    results = []
    for k, y0 in enumerate(candidates):
        try:
            res = fit(problem, y0, max_nfev=screening_nfev, **kwargs)
            results.append({"start": k, "y": res.x, "cost": res.cost, "polished": False, "status": res.status,
                            "optimality": res.optimality})
        except (RuntimeError, ValueError) as err:   # e.g. non-finite residuals at the start point
            results.append({"start": k, "y": y0, "cost": np.inf, "polished": False, "status": f"error: {err}",
                            "optimality": np.inf})
        if verbose:
            print(f"start {k:3d}: cost {results[-1]['cost']:.6g}")
    results.sort(key=lambda r: r["cost"])
    for r in results[:n_polish]:
        if np.isfinite(r["cost"]):
            res = fit(problem, r["y"], max_nfev=polish_nfev, **kwargs)
            r.update(y=res.x, cost=res.cost, polished=True, status=res.status, optimality=res.optimality)
    results.sort(key=lambda r: r["cost"])
    for r in results:
        r["x"] = problem.x_free(r["y"])
    return results


def continuation(problems, n_starts=20, starts=(), n_carry=3, n_polish=3, verbose=False, **kwargs):
    """Continuation over a sequence of problems with the same parameters, from easy to complete data (e.g. first only
    low molalities near 298 K, then all data): the multistart of every stage starts from the `n_carry` best results of
    the previous stage (plus the given starts and, in the first stage, n_starts Sobol points). Returns the results of
    the last stage."""
    results = []
    for k, problem in enumerate(problems):
        carried = [r["y"] for r in results[:n_carry] if np.isfinite(r["cost"])]
        results = multistart(problem, n_starts if k == 0 else 0, list(starts) + carried,
                             n_polish=max(n_polish, len(carried)), verbose=verbose, **kwargs)
        if verbose:
            print(f"stage {k}: best cost {results[0]['cost']:.6g}")
    return results


def waterfall(results, rtol=1e-3):
    """sorted final costs and the number of starts within rtol of the best cost (evidence for the global optimum)"""
    costs = np.array([r["cost"] for r in results])
    best = costs.min()
    return costs, int(np.sum(costs <= best * (1 + rtol)))


def classify(results, par, problem=None, rtol_cost=1e-2, tol_x=1e-2, atol_cost=1e-12, max_nfev=None):
    """Classify the converged (polished) results of a multistart. With `problem`, every polished result is first
    converged again with tight tolerances (xtol = ftol = gtol = 1e-12, at most `max_nfev` evaluations), so that results
    that stopped early in a flat valley are not counted as different minima. Then, relative to the best result:
    - "multimodal": a result with a cost higher by more than rtol_cost (relative, at least atol_cost) and parameters
      different by more than tol_x (in units of TYPICAL): distinct local minima (use `cma_es`);
    - "flat": all costs within rtol_cost, parameters different by more than tol_x: directions the data do not determine;
    - "unimodal": all costs within rtol_cost and parameters within tol_x;
    - "undecided": fewer than two converged starts.
    Returns (class, details)."""
    pol = [r for r in results if r["polished"] and np.isfinite(r["cost"])]
    if len(pol) < 2:
        return "undecided", {"converged": len(pol)}
    if problem is not None:
        for r in pol:
            res = fit(problem, r["y"], max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            r.update(y=res.x, cost=res.cost, status=res.status, optimality=res.optimality, x=problem.x_free(res.x))
        pol.sort(key=lambda r: r["cost"])
    best = pol[0]
    scale = np.array([TYPICAL[n[2]] for n, f in zip(par.names, par.free) if f])
    dx = [float(np.max(np.abs(r["x"] - best["x"]) / scale)) for r in pol]
    higher = [r["cost"] > best["cost"] * (1 + rtol_cost) + atol_cost for r in pol]
    details = {"costs": [r["cost"] for r in pol], "spread": max(dx)}
    if any(h and d > tol_x for h, d in zip(higher, dx)):
        return "multimodal", details
    return ("unimodal" if max(dx) <= tol_x else "flat"), details


def cma_es(problem, y0, sigma0=0.2, max_evals=2000, popsize=None, seed=0, verbose=False):
    """Global search with CMA-ES (pycma) on the objective in the variables y, normalized by the bounds of the optimizer
    (a tenth of their width; variables without bounds: their magnitude); failed evaluations count as infinite cost.
    The result should be polished with `fit`."""
    import cma
    lo, hi = problem.bounds()
    # coordinates z = (y - y0) / width; width: a tenth of the bounds (variables without bounds: their magnitude)
    width = np.where(np.isfinite(hi - lo), (hi - lo) / 10, np.maximum(np.abs(y0), 1.0))
    width = np.where(width > 0, width, 1.0)
    to_y = lambda z: y0 + z * width
    def f(z):
        y = to_y(np.asarray(z))
        if np.any(y < lo) or np.any(y > hi):
            return np.inf
        try:
            c = problem.cost(y)
        except Exception:
            return np.inf
        return c if np.isfinite(c) else np.inf
    opts = {"maxfevals": max_evals, "seed": seed + 1, "verbose": -9 if not verbose else 1,
            "bounds": [list((lo - y0) / width), list((hi - y0) / width)]}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(np.zeros(len(y0)), sigma0, opts)
    es.optimize(f)
    z = es.result.xbest
    return {"y": to_y(np.asarray(z)), "cost": es.result.fbest, "evaluations": es.result.evaluations}


def robust_fit(problem, n_starts=20, starts=(), cma_evals=2000, verbose=False, **kwargs):
    """Multistart; if the converged results are multimodal, CMA-ES from the best result and a final least-squares
    polish. Returns (best result dict, multistart results, classification, CMA-ES result or None)."""
    results = multistart(problem, n_starts, starts, verbose=verbose, **kwargs)
    kind, details = classify(results, problem.par)
    global_result = None
    best = results[0]
    if kind == "multimodal":
        global_result = cma_es(problem, best["y"], max_evals=cma_evals, verbose=verbose)
        res = fit(problem, global_result["y"])
        if res.cost < best["cost"]:
            best = {"start": "cma-es", "y": res.x, "x": problem.x_free(res.x), "cost": res.cost, "polished": True,
                    "status": res.status, "optimality": res.optimality}
    return best, results, (kind, details), global_result


# ---------------------------------------------------------------- diagnostics
def metrics(calc, exp):
    """absolute and relative deviations (calc - exp)"""
    d = np.asarray(calc) - np.asarray(exp)
    r = d / np.asarray(exp)
    return {"n": len(d), "MAD": np.mean(np.abs(d)), "MARD%": 100 * np.mean(np.abs(r)), "RMSE": np.sqrt(np.mean(d ** 2)),
            "RMSRE%": 100 * np.sqrt(np.mean(r ** 2)), "bias": np.mean(d), "rel_bias%": 100 * np.mean(r),
            "max_AD": np.max(np.abs(d)), "max_RD%": 100 * np.max(np.abs(r))}


def block_metrics(objective, par, x):
    """metrics of the measured quantities per block (system, property) for the full parameter vector x; for points with
    "linearized" residuals the model value of the measured quantity (a_w, p, freezing-point depression, boiling-point
    elevation) is measured * (1 + r) (first order in r)"""
    calc, _ = objective.model.evaluate(x, jacobian=False)
    calc = objective.exp + objective.deviations(calc)     # heat capacities without the offset per group (if removed)
    measured = np.array([p.measured for p in objective.model.points])
    linearized = np.array([p.residual == "linearized" for p in objective.model.points])
    calc = np.where(linearized, measured * (1 + (calc - objective.exp) / objective.scale), calc)
    exp = np.where(linearized, measured, objective.exp)
    # heat capacities of isotherms with one point carry no information once the offset is removed: not counted
    sizes = np.append(objective.cp_group_size, 0)        # index -1 (not a heat capacity): size 0
    single = sizes[objective.cp_group] == 1
    out = {}
    for b, name in enumerate(objective.block_names):
        sel = (objective.block == b) & np.isfinite(calc) & ~single
        out[name] = metrics(calc[sel], exp[sel]) if sel.any() else {"n": 0}
        out[name]["failed"] = int(np.sum((objective.block == b) & ~np.isfinite(calc)))
    return out


def identifiability(objective, par, x, problem=None):
    """singular values of the Jacobian (columns scaled by the typical magnitudes TYPICAL) at the full parameter vector x
    and the parameter combinations of the smallest ones; rough standard errors from (J^T J)^-1 * 2 cost / (n - p)
    (Gauss-Newton approximation that ignores the Huber weighting and assumes equal variances of the weighted
    residuals). The pseudo-inverse is taken of the Jacobian with columns scaled to unit norm, so that the cutoff of
    small singular values does not depend on the units of the parameters.
    With `problem` (Problem), everything refers to the variables of the optimizer: for every ion with
    cross_energies, epsilon_k and k_w are replaced by s = sqrt(epsilon_k) ("sqrt_epsilon_k[ion]") and
    u = sqrt(epsilon_k) (1 - k_w) ("u[ion]"); "labels" and "values" give names and values of the variables."""
    par = par.at(x)
    x_free = x[par.free]
    labels = par.labels()
    scale = np.array([TYPICAL[n[2]] for n, f in zip(par.names, par.free) if f])
    if problem is None:
        J = objective.jacobian(x_free, par)
        values = x_free
    else:
        values = problem.to_y(x_free)
        J = problem.jacobian(values)
        labels = list(labels)
        for ie, ik, _ in problem.pairs:
            if ie is not None:
                labels[ie] = labels[ie].replace("epsilon_k[", "sqrt_epsilon_k[")
                scale[ie] = np.sqrt(TYPICAL["epsilon_k"])
            labels[ik] = labels[ik].replace("k_w[", "u[")
            scale[ik] = np.sqrt(TYPICAL["epsilon_k"]) * TYPICAL["k_w"]
    U, s, Vt = np.linalg.svd(J * scale, full_matrices=False)
    weakest = []
    for k in np.argsort(s)[:5]:
        v = Vt[k]
        top = np.argsort(-np.abs(v))[:4]
        weakest.append((s[k], [(labels[i], round(float(v[i]), 3)) for i in top]))
    n, p = J.shape
    norm = np.linalg.norm(J, axis=0)
    norm = np.where(norm > 0, norm, 1.0)
    cov = np.linalg.pinv((J / norm).T @ (J / norm)) / np.outer(norm, norm) * 2 * objective.cost(x_free, par) / max(n - p, 1)
    return {"singular_values": s, "condition": s.max() / max(s.min(), 1e-300), "weakest": weakest,
            "std_error": np.sqrt(np.clip(np.diag(cov), 0.0, None)), "labels": labels, "values": values}


# ---------------------------------------------------------------- symmetries and gauge (MacInnes convention)
def ion_groups(points):
    """groups of ions that are connected through the systems of the data"""
    groups = []
    for ions in {p.ions for p in points}:
        merged = [g for g in groups if g & set(ions)]
        new = set(ions).union(*merged)
        groups = [g for g in groups if g not in merged] + [new]
    return groups


def permittivity_variant(par, variant):
    """ion permittivity parameters: "PT2" (a11mu2 and a2 per ion, Rueben et al. 2024) returns `par` unchanged; "PT1"
    (a11mu2 per ion, a2 of every ion equal to a2 of water) returns a copy with a2 of all ions set to the value of the
    water record and not adjusted"""
    if variant == "PT2":
        return par
    if variant != "PT1":
        raise ValueError(f"permittivity variant {variant!r}: 'PT1' or 'PT2'")
    a2 = par.water["permittivity_record"]["PerturbationTheory"]["correlation_integral_parameter"]
    x, free = par.x0.copy(), par.free.copy()
    for k, (kind, _, p) in enumerate(par.names):
        if kind == "ion" and p == "correlation_integral_parameter":
            x[k], free[k] = a2, False
    return ParameterSet(par.ions, par.records, par.names, x, free, par.water)


def reference_gauge(par, points, fixed):
    """parameter set with the start values mapped by the dispersion symmetry (per group of connected ions) such that
    eps_k of the reference ion equals REFERENCE_EPSILON_K (if given); predictions are unchanged"""
    x = par.x0.copy()
    for group in ion_groups(points):
        for ion in {i for i, q in fixed if q == "epsilon_k" and i in group and i in REFERENCE_EPSILON_K}:
            eps = x[par.index("ion", ion, "epsilon_k")]
            f = np.sqrt(REFERENCE_EPSILON_K[ion] / eps)     # sqrt(eps) -> f sqrt(eps) for the reference ion
            lam = f if par.records[ion]["z"] > 0 else 1.0 / f
            x = dispersion_symmetry(par, x, lam, group)
    return par.at(x)


def gauge_fixed(points, pure, reference=REFERENCE_ION):
    """fixed (ion, parameter) pairs that remove the exact symmetries: both symmetries act separately on every group
    of ions that are connected through the systems of the data; per group, the parameters GAUGE_PARAMETERS of
    `reference` are fixed if it belongs to the group, otherwise those of the anion (z < 0 in the records `pure`) with
    most data points"""
    groups = ion_groups(points)
    count = {}
    for p in points:
        for i in p.ions:
            count[i] = count.get(i, 0) + 1
    anions = {i for p in points for i in p.ions if pure[i]["z"] < 0}
    fixed = []
    for g in groups:
        ref = reference if reference in g else max(sorted(g & anions), key=lambda i: count[i])
        fixed += [(ref, q) for q in GAUGE_PARAMETERS]
    return fixed


def dispersion_symmetry(par, x, lam, ions=None):
    """full vector transformed by the dispersion symmetry with factor lam for `ions` (default: all; must be a whole group
    of connected ions); all model properties unchanged"""
    y = x.copy()
    for ion in par.ions if ions is None else ions:
        f = lam if par.records[ion]["z"] > 0 else 1.0 / lam
        y[par.index("ion", ion, "epsilon_k")] *= f ** 2
        k = par.index("ion", ion, "k_w")
        y[k] = 1.0 - (1.0 - y[k]) / f
    return y


def permittivity_gauge(par, x, c_dipole, c_correlation):
    """full vector with (dipole_scaling, correlation_integral_parameter) of all ions shifted by (c_dipole, c_correlation) z_i
    (all electroneutral properties unchanged)"""
    y = x.copy()
    for ion in par.ions:
        z = par.records[ion]["z"]
        y[par.index("ion", ion, "dipole_scaling")] += c_dipole * z
        y[par.index("ion", ion, "correlation_integral_parameter")] += c_correlation * z
    return y



def macinnes_gauge(par, x, T=298.15, p=1e5, molalities=np.linspace(0.1, 4.0, 20), variant="dh",
                   shift=("dipole_scaling",), msa_packing="ions"):
    """Shift the permittivity parameters `shift` of all ions by c z_i so that ln gamma(K+) = ln gamma(Cl-) in aqueous KCl
    (least squares over `molalities` at T). All electroneutral properties are unchanged. The convention holds only
    approximately with constant shifts; by default only dipole_scaling is shifted (the second constant is nearly
    undetermined by the convention), correlation_integral_parameter keeps the value of the reference ion.
    Returns (shifted full vector, shifts, max |ln gamma(K+) - ln gamma(Cl-)| after the shift)."""
    if not {"potassium ion", "chloride ion"} <= set(par.ions):
        raise ValueError("MacInnes convention needs K+ and Cl-")
    ions = ("chloride ion", "potassium ion")
    pts = [ed.Point(0, "", "KCl", "", "", "", T, p, {"KCl": m}) for m in molalities]
    model = Model(par, pts, variant, msa_packing)

    def shifted(c):
        c = dict(zip(shift, c))
        return permittivity_gauge(par, x, c.get("dipole_scaling", 0.0), c.get("correlation_integral_parameter", 0.0))

    def diff(c):
        y = shifted(c)
        params = model._eos_parameters(ions, y)
        rows = np.array([[T, p] + Model._moles(ions, pt.composition) for pt in pts])
        d_cat, d_an = par.born_delta(y)
        v, _, _ = feos.Property.electrolyte_properties_derivatives(
            params, [], rows, solvent=0, epcpsaft_variant=variant, msa_packing=msa_packing,
            born_delta_cation=d_cat, born_delta_anion=d_an)
        return v["ln_gamma"][:, 2] - v["ln_gamma"][:, 1]   # ln gamma(K+) - ln gamma(Cl-)

    c = least_squares(diff, np.zeros(len(shift))).x
    return shifted(c), c, float(np.max(np.abs(diff(c))))


# ---------------------------------------------------------------- output (feos JSON format)
def write_parameters(par, x, directory, name="ions"):
    """ion records (<name>.json) and water-ion k_ij (<name>_binary.json) in feos format; Born corrections and Cp offsets
    in <name>_options.json"""
    os.makedirs(directory, exist_ok=True)
    pure, binary = par.ion_records(x)
    json.dump(pure, open(os.path.join(directory, f"{name}.json"), "w"), indent=4)
    json.dump(binary, open(os.path.join(directory, f"{name}_binary.json"), "w"), indent=4)
    extra = dict(zip(GLOBAL_PARAMETERS, map(float, par.born_delta(x))))
    extra["Cp_offset"] = {n[1]: float(x[i]) for i, n in enumerate(par.names) if n[0] == "Cp_offset"}
    json.dump(extra, open(os.path.join(directory, f"{name}_options.json"), "w"), indent=4)
