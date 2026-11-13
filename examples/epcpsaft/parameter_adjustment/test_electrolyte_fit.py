"""Tests of the ion-parameter fit (data reduction, exact derivatives, symmetries, parameter recovery, optimizers).

Needs the dummy data (~/ePCP_dummy_data, README there): set EPCPSAFT_DUMMY_DATA to its folder. The second set
(data_epcpsaft/, generated with the ePCP-SAFT truth model params_epcpsaft/, Debye-Hückel) is used for exactness and
recovery tests, the first set (data/, ePC-SAFT truth model) for a smoke test.

    EPCPSAFT_DUMMY_DATA=~/ePCP_dummy_data pytest test_electrolyte_fit.py
"""
import json
import os
import tempfile

import feos
import numpy as np
import pandas as pd
import pytest

import electrolyte_data as ed
import electrolyte_fit as ef

ROOT = os.environ.get("EPCPSAFT_DUMMY_DATA")
needs_data = pytest.mark.skipif(ROOT is None, reason="EPCPSAFT_DUMMY_DATA not set")


def setup(systems=None, value_column="value_true", data="data_epcpsaft", variant="dh", reference=False,
          msa_packing="ions"):
    """parameter set (truth values of the second dummy set) and objective; reference=True maps the start values to
    the reference gauge (eps_k of Cl- = 170 K)"""
    df = ed.read(os.path.join(ROOT, data))
    if systems is not None:
        df = df[df.system.isin(systems)]
    points, _ = ed.reduce(df, value_column=value_column)
    params = os.path.join(ROOT, "params_epcpsaft")
    pure, k_w = ef.read_records(os.path.join(params, "truth_pure.json"), os.path.join(params, "truth_binary.json"))
    options = json.load(open(os.path.join(params, "options.json")))
    ions = sorted({i for p in points for i in p.ions})
    cp = sorted({ef.cp_block(p) for p in points if p.target == "Cp_phi"})
    fixed = ef.gauge_fixed(points, pure)
    par = ef.ParameterSet.new(ions, pure, k_w, ef.water_record(), (options["born_delta_cation"], options["born_delta_anion"]),
                              cp, fixed=fixed)
    if reference:
        par = ef.reference_gauge(par, points, fixed)
    return par, ef.Objective(ef.Model(par, points, variant, msa_packing, cp_model="offset"))


def test_huber():
    """0.5 psi^2 is the Huber loss; psi' is its derivative"""
    delta = 0.05
    r = np.linspace(-0.3, 0.3, 601)
    psi, dpsi = ef.huber(r, delta)
    loss = np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))
    assert np.allclose(0.5 * psi ** 2, loss, rtol=1e-12, atol=1e-15)
    h = 1e-7
    fd = (ef.huber(r + h, delta)[0] - ef.huber(r - h, delta)[0]) / (2 * h)
    mask = np.abs(np.abs(r) - delta) > 1e-6
    assert np.allclose(dpsi[mask], fd[mask], rtol=1e-6)


@needs_data
def test_binding_checks():
    """invalid input of feos.Property.electrolyte_properties_derivatives raises errors instead of failing rows"""
    par, obj = setup(["NaCl"])
    params = obj.model._eos_parameters(("chloride ion", "sodium ion"), par.x0)
    row = np.array([[298.15, 1e5, 1 / ed.M_W, 1.0, 1.0]])
    f = feos.Property.electrolyte_properties_derivatives
    v, g, ok = f(params, [("sigma", [1]), ("k_ij", [0, 2])], row)
    assert ok[0] and g["ln_gamma"].shape == (1, 3, 2) and v["ln_gamma"].shape == (1, 3)
    for spec in ([("k_ij", [1, 1])], [("mu", [1])], [("sigma", [9])], [("sigma", [1])] * 2,
                 [("k_ij", [0, 1]), ("k_ij", [1, 0])], [("unknown", [1])]):
        with pytest.raises(ValueError):
            f(params, spec, row)
    with pytest.raises(ValueError):
        f(params, [], np.array([[298.15, 1e5, 1 / ed.M_W, 1.0, 0.5]]))       # not electroneutral
    with pytest.raises(ValueError):
        f(params, [], row, solvent=3)


@needs_data
def test_truth_residuals():
    """the model reproduces the noise-free dummy data of its own truth model (independent implementation of all
    properties in the generator, ~/ePCP_dummy_data/model.py); freezing and boiling points are limited by the
    generator's bisection tolerance (1e-6 K; compared as deviation of the temperature, since depressions and elevations
    go down to 3e-5 K), the other points by the precision of the density iterations (osmotic coefficients at
    1e-3 mol/kg: relative 2e-6, i.e. 4e-11 in ln a_w)"""
    par, obj = setup()
    r, _ = obj.relative_residuals(par.x0)
    assert np.all(np.isfinite(r))
    pts = obj.model.points
    temperature = np.array([p.property in ("freezing_temperature", "boiling_temperature") for p in pts])
    theta = np.array([p.measured for p in pts])
    assert np.max(np.abs(r[~temperature])) < 1e-5
    assert np.max(np.abs(r * theta)[temperature]) < 1e-6


@needs_data
def test_linearized_residuals():
    """colligative residuals are the relative deviations of the measured quantity to first order: a model ln a_w
    shifted by dl gives r = dl / scale, compared with the exact relative change of the measured quantity"""
    import iapws
    dl = 1e-4
    df = ed.read(os.path.join(ROOT, "data_epcpsaft"))
    for prop in ("vapor_pressure", "freezing_temperature", "boiling_temperature", "water_activity"):
        rows = df[df.property == prop].iloc[[5]]
        (pt,), _ = ed.reduce(rows, value_column="value_true")
        row = rows.iloc[0]
        r = dl / pt.scale
        v = row["value_true"]
        if prop == "vapor_pressure":
            exact = iapws.vapor_pressure(pt.T, np.exp(pt.value + dl)) / v - 1
        elif prop == "water_activity":
            exact = np.exp(dl) - 1
        elif prop == "freezing_temperature":
            from scipy.optimize import brentq
            t = brentq(lambda t: ed._ln_aw_ice(t) - (pt.value + dl), v - 1, v + 1)    # new freezing point
            exact = (ed.T_MELT - t) / (ed.T_MELT - v) - 1
        else:
            from scipy.optimize import brentq
            t = brentq(lambda t: np.log(iapws.water_activity(t, 101325.0)) - (pt.value + dl), v - 1, v + 1)
            t0 = iapws.saturation_temperature(101325.0)
            exact = (t - t0) / (v - t0) - 1
        assert r == pytest.approx(exact, rel=0.05, abs=1e-6), prop


@needs_data
@pytest.mark.parametrize("variant,msa_packing", [("dh", "ions"), ("msa", "ions"), ("msa", "total")])
def test_jacobian(variant, msa_packing):
    """exact Jacobian of the residual vector (all targets, mixtures, Cp offsets; Debye-Hückel and MSA) against
    4th-order central differences at a perturbed parameter vector. The Huber threshold is set so large that all
    residuals are in the quadratic branch (finite differences across the kink of the Huber loss are not accurate);
    the Huber transformation is tested in test_huber."""
    systems = ["NaCl", "HCl", "NiSO4", "(NH4)2SO4", "K2CO3", "Li2SO4+Na2SO4", "KCl"]
    par, obj = setup(systems, variant=variant, msa_packing=msa_packing)
    obj = ef.Objective(obj.model, delta=1e3)
    x = par.x0[par.free]
    rng = np.random.default_rng(0)
    xp = x * (1 + 0.03 * rng.uniform(-1, 1, x.size)) + 0.01 * rng.uniform(-1, 1, x.size)
    J = obj.jacobian(xp, par)
    r0, _ = obj.relative_residuals(par.full(xp))

    def f(y):
        r, _ = obj.relative_residuals(par.full(y))
        return obj.residuals(y, par), np.isfinite(r)

    def error(k, h):
        e = np.zeros(x.size)
        e[k] = h
        (f1, c1), (f2, c2), (f3, c3), (f4, c4) = f(xp + e), f(xp - e), f(xp + 2 * e), f(xp - 2 * e)
        fd = (8 * (f1 - f2) - (f3 - f4)) / (12 * h)
        # rows converged in all evaluations; rows with residual scales below 1e-3 (freezing and boiling points with
        # depressions or elevations below about 0.1 K) are excluded: they amplify the convergence tolerance of the
        # density iteration beyond the accuracy of the finite differences
        ok = c1 & c2 & c3 & c4 & np.isfinite(r0) & (np.abs(obj.scale) > 1e-3)
        # relative to the column, with a floor for columns that are tiny compared to the whole Jacobian
        return np.max(np.abs(J[ok, k] - fd[ok])) / (np.max(np.abs(fd[ok])) + 1e-4 * np.max(np.abs(J)))

    # finite differences are limited by truncation (large h) and by the convergence tolerance of the density
    # iteration (small h); further steps are tried only if the first does not agree (a wrong derivative fails at all)
    for k, name in enumerate(par.labels()):
        typical = ef.TYPICAL[par.names[np.flatnonzero(par.free)[k]][2]]
        errors = []
        for step in (1e-2, 1e-1, 3e-2, 3e-3, 1e-3):
            errors.append(error(k, step * typical))
            if errors[-1] < 1e-4:
                break
        assert min(errors) < 1e-4, f"{name}: {min(errors):.2e}"


@needs_data
def test_problem_reparametrization():
    """cross-energy variables: round trip and chain rule of the Jacobian"""
    par, obj = setup(["NaCl", "MgSO4"], reference=True)
    problem = ef.Problem(obj, par, cross_energies=True)
    x = par.x0[par.free]
    y = problem.to_y(x)
    assert np.allclose(problem.x_free(y), x, rtol=1e-14)
    obj_q = ef.Objective(obj.model, delta=1e3)
    problem = ef.Problem(obj_q, par, cross_energies=True)
    J = problem.jacobian(y)
    ok = np.abs(obj_q.scale) > 1e-3      # as in test_jacobian: dilute freezing and boiling points are excluded
    for k in range(len(y)):
        h = 1e-4 * max(abs(y[k]), 1.0)
        e = np.zeros(len(y))
        e[k] = h
        fd = (problem.residuals(y + e) - problem.residuals(y - e)) / (2 * h)
        assert np.max(np.abs(J[ok, k] - fd[ok])) <= 1e-4 * np.max(np.abs(J)) + 1e-5 * np.max(np.abs(fd)), k


@needs_data
def test_symmetries():
    """permittivity gauge and dispersion symmetry leave all targets unchanged (also with MSA); the reference gauge
    maps eps_k of Cl- to 170 K without changing any target. At extreme states (MgSO4 12 mol/kg, 473 K) the rounding
    noise of the pressure (2e-5 Pa) exceeds the tolerance of the density iteration of feos (1e-12 in reduced units,
    1.4e-5 Pa), so that a point can converge for one of two equivalent parameter sets only: at most two such rows"""
    for variant, packing in (("dh", "ions"), ("msa", "ions"), ("msa", "total")):
        par, obj = setup(["NaCl", "KCl", "MgSO4", "Li2SO4+Na2SO4", "NiSO4"], variant=variant, msa_packing=packing)
        a, _ = obj.model.evaluate(par.x0, jacobian=False)
        for y in (ef.permittivity_gauge(par, par.x0, 1.7, -0.08), ef.dispersion_symmetry(par, par.x0, 1.3)):
            b, _ = obj.model.evaluate(y, jacobian=False)
            both = np.isfinite(a) & np.isfinite(b)
            assert both.sum() >= len(a) - 2
            assert np.max(np.abs(b - a)[both] / obj.scale[both]) < 1e-7
    par0 = par.at(ef.dispersion_symmetry(par, par.x0, 1.2))      # start values off the reference gauge
    ref = ef.reference_gauge(par0, obj.model.points, ef.gauge_fixed(obj.model.points, par.records))
    assert ref.x0[ref.index("ion", "chloride ion", "epsilon_k")] == pytest.approx(170.0, rel=1e-12)
    b, _ = obj.model.evaluate(ref.x0, jacobian=False)
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() >= len(a) - 2
    assert np.max(np.abs(b - a)[both] / obj.scale[both]) < 1e-7


@needs_data
def test_permittivity_variant():
    """PT1: a2 of all ions equal to a2 of water and not adjusted; PT2 unchanged; with PT1 the a2 directions of the
    permittivity gauge are removed and the Jacobian has no null space"""
    par, obj = setup(["NaCl", "KCl", "MgSO4"])
    assert ef.permittivity_variant(par, "PT2") is par
    pt1 = ef.permittivity_variant(par, "PT1")
    a2 = par.water["permittivity_record"]["PerturbationTheory"]["correlation_integral_parameter"]
    k = [i for i, n in enumerate(par.names) if n[2] == "correlation_integral_parameter"]
    assert np.all(pt1.x0[k] == a2) and not pt1.free[k].any()
    assert pt1.free.sum() == par.free.sum() - (len(k) - 2)      # two reference anions had a2 fixed already
    values, _ = obj.model.evaluate(pt1.x0, jacobian=False)
    assert np.all(np.isfinite(values))
    s = ef.identifiability(obj, pt1, pt1.x0)["singular_values"]
    assert s.min() / s.max() > 1e-8
    with pytest.raises(ValueError):
        ef.permittivity_variant(par, "PT3")


def test_ideal_gas_cp():
    """ideal-gas heat capacities of ions: 5/2 R for monatomic ions; polyatomic ions between the rigid-rotor value
    (5/2 R + 3/2 R, or + R if linear) and the classical limit with all vibrations (one R per mode)"""
    R = ef.R
    assert ef.ideal_gas_cp("sodium ion", 298.15) == pytest.approx(2.5 * R, rel=1e-15)
    for ion, (linear, modes) in ef.POLYATOMIC_IONS.items():
        rigid = R * (2.5 + (1.0 if linear else 1.5))
        n_modes = sum(g for _, g in modes)
        assert rigid < ef.ideal_gas_cp(ion, 298.15) < ef.ideal_gas_cp(ion, 600.0) < rigid + n_modes * R
        assert ef.ideal_gas_cp(ion, 1e5) == pytest.approx(rigid + n_modes * R, rel=1e-3)
    with pytest.raises(ValueError):
        ef.ideal_gas_cp("unknown ion", 298.15)
    # electronic contribution: zero for single-level and closed-shell ions, two-level system for Cu2+
    assert ef.electronic_cp("sodium ion", 298.15) == 0.0 and ef.electronic_cp("manganese ion", 298.15) == 0.0
    x = ef.C2_CM_K * 2071.69 / 298.15
    two_level = R * (4 / 6) * x ** 2 * np.exp(-x) / (1 + (4 / 6) * np.exp(-x)) ** 2
    assert ef.electronic_cp("copper ion", 298.15) == pytest.approx(two_level, rel=1e-12)
    assert ef.ideal_gas_cp("cobalt ion", 298.15, electronic=True) - ef.ideal_gas_cp("cobalt ion", 298.15) == \
        pytest.approx(ef.electronic_cp("cobalt ion", 298.15), rel=1e-15)


@needs_data
def test_cp_models_and_bounds():
    """cp_model "ideal_gas" adds sum_i m_i c_p,i^ig / m_salt to the residual value of "residual"; the optimizer bounds
    are only the physical ones (sigma, eps_k, cross-energy variables >= 0, k_w <= 1)"""
    par, obj = setup(["NaCl", "Na2SO4"])
    pts = [p for p in obj.model.points if p.target == "Cp_phi"][:20]
    v_res, _ = ef.Model(par, pts, cp_model="residual").evaluate(par.x0, jacobian=False)
    v_ig, _ = ef.Model(par, pts, cp_model="ideal_gas").evaluate(par.x0, jacobian=False)
    expected = [sum(m * ef.ideal_gas_cp(i, p.T) for i, m in ed.ions_of(p.composition).items())
                / sum(p.composition.values()) for p in pts]
    assert np.allclose(v_ig - v_res, expected, rtol=1e-12)
    with pytest.raises(ValueError):
        ef.Model(par, pts, cp_model="other")
    lo, hi = par.bounds()
    labels = par.labels()
    for l, a, b in zip(labels, lo, hi):
        kind = l.split("[")[0]
        assert (a, b) == {"sigma": (0.0, np.inf), "epsilon_k": (0.0, np.inf), "k_w": (-np.inf, 1.0)}.get(
            kind, (-np.inf, np.inf))
    plo, phi_ = ef.Problem(obj, par, cross_energies=True).bounds()
    ions = [i for i, l in enumerate(labels) if l.split("[")[0] in ("epsilon_k", "k_w")]
    assert np.all(plo[ions] == 0.0) and np.all(np.isinf(phi_[ions]))


def test_data_module_salts_split_electrolyte():
    """salts from _salts.json, filter by split, mean ionic activity coefficient of the named electrolyte"""
    with tempfile.TemporaryDirectory() as tmp:
        json.dump({"XCl": {"sodium ion": 1, "chloride ion": 1}}, open(os.path.join(tmp, "_salts.json"), "w"))
        rows = [dict(dataset_id="a", system="NaCl+XCl", property="mean_ionic_activity_coefficient", T_K=298.15,
                     p_Pa=1e5, salt_1="NaCl", m_1=0.5, salt_2="XCl", m_2=0.5, m_1_final=np.nan, m_2_final=np.nan,
                     value=0.7, electrolyte="XCl", split="train"),
                dict(dataset_id="b", system="NaCl", property="osmotic_coefficient", T_K=298.15, p_Pa=1e5,
                     salt_1="NaCl", m_1=1.0, salt_2=np.nan, m_2=np.nan, m_1_final=np.nan, m_2_final=np.nan,
                     value=0.93, electrolyte="", split="validation")]
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "data.csv"), index=False)
        df = ed.read(tmp)
        assert ed.SALTS["XCl"] == {"sodium ion": 1, "chloride ion": 1}
        pts, _ = ed.reduce(df, splits=["train"])
        assert len(pts) == 1 and pts[0].electrolyte == "XCl" and pts[0].split == "train"
        pts, _ = ed.reduce(df)
        assert {p.split for p in pts} == {"train", "validation"}
        json.dump({"XCl": {"potassium ion": 1, "chloride ion": 1}}, open(os.path.join(tmp, "_salts.json"), "w"))
        with pytest.raises(ValueError):
            ed.read(tmp)
        del ed.SALTS["XCl"]


def _sulfuric_acid(ln_k0=None, n_points=40, m_max=6.0):
    """dummy H2SO4 data up to m_max (fully dissociated truth) with an HSO4- record (sulfate values, charge -1, for
    tests) and the bisulfate equilibrium (ln K0 of Knopf et al. 2003 or `ln_k0`)"""
    df = ed.read(os.path.join(ROOT, "data_epcpsaft"))
    df = df[(df.system == "H2SO4") & (df.m_1 <= m_max)]
    points, _ = ed.reduce(df, value_column="value_true")
    rng = np.random.default_rng(1)
    points = [points[k] for k in sorted(rng.choice(len(points), n_points, replace=False))]
    params = os.path.join(ROOT, "params_epcpsaft")
    pure, k_w = ef.read_records(os.path.join(params, "truth_pure.json"), os.path.join(params, "truth_binary.json"))
    hso4 = dict(pure["sulfate"], identifier={"name": "hydrogen-sulfate"}, z=-1, molarweight=97.0705)
    pure["hydrogen-sulfate"] = hso4
    reactions = [dict(ef.REACTIONS[0], **({} if ln_k0 is None else {"ln_k0": ln_k0}))]
    ions = ef.model_ions(points, reactions)
    par = ef.ParameterSet.new(ions, pure, k_w, ef.water_record())
    return par, points, reactions


@needs_data
def test_reactions():
    """bisulfate equilibrium: (1) species, basis amounts and feos input; (2) with ln K = 15 (dissociation practically
    complete; up to 2 mol/kg, at higher molalities the activity coefficients of the dummy model are large enough that
    HSO4- forms also with ln K = 15) all targets equal those without reaction (heat capacity: central difference in T);
    (3) up to 6 mol/kg with the
    equilibrium constant of Knopf et al. (2003) the Jacobian agrees with central differences"""
    par, points, reactions = _sulfuric_acid(ln_k0=15.0, n_points=120, m_max=2.0)
    assert ef.system_species(("hydronium", "sulfate"), reactions) == ("hydrogen-sulfate", "hydronium", "sulfate")
    assert ef.system_species(("chloride ion", "hydronium"), reactions) == ("chloride ion", "hydronium")
    sp = ("hydrogen-sulfate", "hydronium", "sodium ion", "sulfate")
    assert ef.basis_moles(sp, {"NaHSO4": 1.0}, reactions)[1:] == [0.0, 1.0, 1.0, 1.0] if "NaHSO4" in ed.SALTS else True
    assert ef.feos_reactions(("hydrogen-sulfate", "hydronium", "sulfate"), reactions)[0][0] == [0.0, -1.0, 1.0, 1.0]
    a, _ = ef.Model(par, points, cp_model="residual").evaluate(par.x0, jacobian=False)
    b, _ = ef.Model(par, points, cp_model="residual", reactions=reactions).evaluate(par.x0, jacobian=False)
    cp = np.array([p.target == "Cp_phi" for p in points])
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.allclose(b[~cp], a[~cp], rtol=1e-5, atol=1e-9)
    assert np.allclose(b[cp], a[cp], rtol=1e-3, atol=1e-2)

    par, points, reactions = _sulfuric_acid()
    model = ef.Model(par, points, cp_model="residual", reactions=reactions)
    v, jac = model.evaluate(par.x0, jacobian=True)
    assert np.all(np.isfinite(v))
    for ion, name in [("hydrogen-sulfate", "sigma"), ("hydrogen-sulfate", "epsilon_k"), ("hydrogen-sulfate", "k_w"),
                      ("hydrogen-sulfate", "dipole_scaling"), ("sulfate", "sigma")]:
        k = par.index("ion", ion, name)
        f = lambda dx: model.evaluate(par.x0 + dx * np.eye(len(par.x0))[k], jacobian=False)[0]
        tol = np.where([p.target == "Cp_phi" for p in points], 1e-3, 2e-4)
        # differences are limited by truncation (large steps, high temperatures) and by the noise of ln a_w (small
        # steps, osmotic coefficients at low molality): a row agrees if one of the steps agrees
        ok = np.zeros(len(points), bool)
        for h in (1e-2 * ef.TYPICAL[name], 3e-2 * ef.TYPICAL[name]):
            fd = (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)
            ok |= np.abs(jac[:, k] - fd) <= tol * np.abs(fd) + 1e-7 * (1 + np.abs(v))
        assert np.all(ok), (ion, name, [points[i].target for i in np.flatnonzero(~ok)])


@needs_data
def test_gauge_groups():
    """one reference ion per group of ions connected through the data; with it the Jacobian has no null space"""
    par, obj = setup(["NaCl", "KCl", "MgSO4"])
    assert sorted({ion for ion, _ in ef.gauge_fixed(obj.model.points, par.records)}) == ["chloride ion", "sulfate"]
    s = ef.identifiability(obj, par, par.x0)["singular_values"]
    assert s.min() / s.max() > 1e-8


@needs_data
def test_macinnes_gauge():
    """the MacInnes shift changes no target and reduces |ln gamma(K+) - ln gamma(Cl-)| in KCl(aq)"""
    par, obj = setup(["NaCl", "KCl"])
    y, c, after = ef.macinnes_gauge(par, par.x0)
    a, _ = obj.model.evaluate(par.x0, jacobian=False)
    b, _ = obj.model.evaluate(y, jacobian=False)
    assert np.max(np.abs(b - a) / obj.scale) < 1e-7
    _, c2, after2 = ef.macinnes_gauge(par, y)
    assert np.all(np.abs(c2) < 1e-4) and after2 == pytest.approx(after, rel=1e-4)   # idempotent
    assert after < 0.05


@needs_data
@pytest.mark.parametrize("cross_energies", [False, True])
def test_recovery(cross_energies):
    """noise-free data of the truth model: from a start 5 % away the fit returns to the truth (cost and parameters),
    with both parametrizations"""
    par, obj = setup(["NaCl", "KCl", "LiCl", "Li2SO4", "Na2SO4", "K2SO4"])
    problem = ef.Problem(obj, par, cross_energies)
    truth = par.x0[par.free].copy()
    rng = np.random.default_rng(2)
    y0 = problem.to_y(truth * (1 + 0.05 * rng.uniform(-1, 1, truth.size)))
    res = ef.fit(problem, y0, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert res.cost < 1e-9
    x = problem.x_free(res.x)
    ion = [k for k, n in enumerate(par.labels()) if not n.startswith("Cp_offset")]
    err = np.abs(x - truth) / np.maximum(np.abs(truth), 1e-2)
    assert np.max(err[ion]) < 1e-3, dict(zip(np.array(par.labels())[ion], err[ion]))


@needs_data
def test_msa_fit():
    """the fit runs with the MSA variant (data of the Debye-Hückel truth: cost decreases)"""
    par, obj = setup(["NaCl", "KCl"], value_column="value", variant="msa")
    problem = ef.Problem(obj, par)
    y0 = problem.to_y(par.x0[par.free])
    res = ef.fit(problem, y0, max_nfev=5)
    assert np.isfinite(res.cost) and res.cost < problem.cost(y0)


@needs_data
def test_multistart_continuation_classify():
    """multistart and continuation return sorted results; classify recognizes converged starts at the same optimum"""
    df = ed.read(os.path.join(ROOT, "data_epcpsaft"))
    df = df[df.system.isin(["NaCl"])]
    par, obj = setup(["NaCl"])
    problem = ef.Problem(obj, par)
    easy, _ = ed.reduce(df[(df.m_1 <= 1.0) & (df.T_K.fillna(298.15) <= 323.15)], value_column="value_true")
    stage = ef.Problem(ef.Objective(ef.Model(par, easy, cp_model="offset")), par)
    y_truth = problem.to_y(par.x0[par.free])
    starts = [y_truth * (1 + 0.02 * k) for k in (1, -1, 2)]
    results = ef.continuation([stage, problem], n_starts=2, starts=starts, n_polish=3, screening_nfev=5)
    costs, n_best = ef.waterfall(results)
    assert np.all(np.diff(costs) >= 0) and np.isfinite(costs[0]) and n_best >= 1
    kind, details = ef.classify(results, par)
    assert kind in ("unimodal", "flat", "multimodal")
    n = par.free.sum()
    fake = lambda costs, shifts: [{"polished": True, "cost": c, "x": np.full(n, s)} for c, s in zip(costs, shifts)]
    big = 10 * max(ef.TYPICAL.values())
    assert ef.classify(fake((1.0, 1.0, 2.0), (0, 0, big)), par)[0] == "multimodal"
    assert ef.classify(fake((1.0, 1.005, 1.0), (0, big, 0)), par)[0] == "flat"     # within 1 %: flat valley
    assert ef.classify(fake((1.0, 1.0, 2.0), (0, 0, 0)), par)[0] == "unimodal"
    kind, _ = ef.classify(results, par, problem=problem)                           # with tight re-polish
    assert kind in ("unimodal", "flat", "multimodal")


@needs_data
def test_cma_es():
    """CMA-ES fallback reduces the cost of a perturbed start (few evaluations)"""
    par, obj = setup(["NaCl"])
    problem = ef.Problem(obj, par)
    y0 = problem.to_y(par.x0[par.free] * 1.05)
    res = ef.cma_es(problem, y0, sigma0=0.05, max_evals=60)
    assert res["cost"] <= problem.cost(y0) and res["evaluations"] <= 60 + 20   # whole last generation


@needs_data
def test_smoke_first_dummy_set():
    """first dummy set (ePC-SAFT truth, different water model): a few iterations run and reduce the cost"""
    par, obj = setup(["NaCl", "KCl", "MgSO4"], value_column="value", data="data", reference=True)
    problem = ef.Problem(obj, par)
    y0 = problem.to_y(par.x0[par.free])
    res = ef.fit(problem, y0, max_nfev=5)
    assert np.isfinite(res.cost) and res.cost < problem.cost(y0)
    assert np.all(np.isfinite(problem.jacobian(res.x)))


@needs_data
def test_write_parameters():
    """written records are valid feos input and reproduce the parameter vector"""
    par, obj = setup(["NaCl"])
    with tempfile.TemporaryDirectory() as d:
        ef.write_parameters(par, par.x0, d)
        pure, k_w = ef.read_records(os.path.join(d, "ions.json"), os.path.join(d, "ions_binary.json"))
        options = json.load(open(os.path.join(d, "ions_options.json")))
        par2 = ef.ParameterSet.new(par.ions, pure, k_w, par.water, (options["born_delta_cation"], options["born_delta_anion"]),
                                   list(options["Cp_offset"]))
        for block, value in options["Cp_offset"].items():
            par2.x0[par2.index("Cp_offset", block, "Cp_offset")] = value
        assert np.allclose(par2.x0, par.x0)
        a, _ = obj.model.evaluate(par.x0, jacobian=False)
        b, _ = ef.Model(par2, obj.model.points, cp_model="offset").evaluate(par2.x0, jacobian=False)
        assert np.allclose(a, b, rtol=1e-12)
