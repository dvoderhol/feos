use crate::DensityInitialization::{InitialDensity, Liquid};
use crate::{Contributions, FeosError, FeosResult, ReferenceSystem, Residual, State};
use nalgebra::{DMatrix, DVector, Dyn};
use num_dual::{DualNum, DualStruct};
use quantity::{
    Density, Energy, Entropy, JOULE, KELVIN, MOL, Moles, PASCAL, Pressure, RGAS, Temperature,
    Volume,
};

/// Chemical reaction equilibrium in an electrolyte solution,
/// $\sum_i\nu_i\ln a_i = \ln K(T)$, with the activities of the solutes in the molality scale
/// (standard state: hypothetical ideal solution of 1 mol/kg in the pure solvent) and of the
/// solvent relative to the pure solvent.
///
/// The temperature dependence of the equilibrium constant follows from a reaction heat capacity
/// that is linear in temperature (Knopf et al. 2003, eq. 16):
/// $\ln K(T) = \ln K_0 - \left[\left(\frac{\Delta H_0}{R} - \frac{\Delta c_{p,0}}{R}T_0 +
/// \frac{1}{2}\frac{c'}{R}T_0^2\right)\left(\frac{1}{T}-\frac{1}{T_0}\right) -
/// \left(\frac{\Delta c_{p,0}}{R} - \frac{c'}{R}T_0\right)\ln\frac{T}{T_0} -
/// \frac{1}{2}\frac{c'}{R}(T - T_0)\right]$,
/// i.e. $\Delta H^\circ(T) = \Delta H_0 + \Delta c_{p,0}(T-T_0) + \frac{1}{2}c'(T-T_0)^2$.
#[derive(Clone, Debug)]
pub struct ReactionEquilibrium {
    /// Stoichiometric coefficients of all components (products positive).
    pub stoichiometry: DVector<f64>,
    /// $\ln K$ at `t0`.
    pub ln_k0: f64,
    /// Reference temperature in K.
    pub t0: f64,
    /// Standard reaction enthalpy at `t0` divided by R, in K.
    pub dh0: f64,
    /// Standard reaction heat capacity at `t0` divided by R.
    pub dcp0: f64,
    /// Temperature derivative of the standard reaction heat capacity divided by R, in 1/K.
    pub dcp_dt: f64,
}

impl ReactionEquilibrium {
    pub fn new(
        stoichiometry: DVector<f64>,
        ln_k0: f64,
        t0: f64,
        dh0: f64,
        dcp0: f64,
        dcp_dt: f64,
    ) -> Self {
        Self {
            stoichiometry,
            ln_k0,
            t0,
            dh0,
            dcp0,
            dcp_dt,
        }
    }

    /// Natural logarithm of the equilibrium constant at the temperature `t` (K).
    pub fn ln_k(&self, t: f64) -> f64 {
        let (t0, h, c, d) = (self.t0, self.dh0, self.dcp0, self.dcp_dt);
        self.ln_k0
            - ((h - c * t0 + 0.5 * d * t0 * t0) * (1.0 / t - 1.0 / t0)
                - (c - d * t0) * (t / t0).ln()
                - 0.5 * d * (t - t0))
    }

    /// Standard reaction enthalpy at the temperature `t` (K) divided by R, in K.
    pub fn reaction_enthalpy(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        self.dh0 + self.dcp0 * dt + 0.5 * self.dcp_dt * dt * dt
    }
}

/// Properties of an electrolyte solution relative to the pure solvent at the same temperature
/// and pressure.
///
/// The solution consists of the amounts `moles`, usually per kg of solvent. Differences of
/// extensive properties refer to the pure solvent with the amount `moles[solvent]`, i.e., they
/// are per kg of solvent if `moles` is given per kg of solvent. Residual properties refer to the
/// ideal gas at the same temperature, pressure and composition; ideal-gas contributions of the
/// solutes are not included. With reaction equilibria, all properties refer to the equilibrium
/// composition.
pub struct ElectrolyteProperties<D> {
    /// Natural logarithm of the solvent activity $\ln a_s = \ln x_s + \ln\varphi_s - \ln\varphi_s^0$
    /// (fugacity coefficients referred to the given pressure, see `ln_phi_at`).
    pub ln_solvent_activity: D,
    /// Natural logarithm of the activity coefficients in the mole-fraction scale: solutes with
    /// the reference state of infinite dilution in the pure solvent,
    /// $\ln\gamma_i^{*} = \ln\varphi_i - \ln\varphi_i^\infty$; solvent with the pure solvent as
    /// reference, $\ln\gamma_s = \ln\varphi_s - \ln\varphi_s^0$. Molality scale:
    /// $\ln\gamma_i^{*,m} = \ln\gamma_i^{*} + \ln x_s$.
    pub ln_gamma: DVector<D>,
    /// Mole fraction of the solvent.
    pub solvent_molefrac: f64,
    /// Amounts of all components (in chemical equilibrium if reactions are given).
    pub moles: DVector<D>,
    /// Extents of the reactions in mol.
    pub extents: DVector<D>,
    /// Volume of the solution minus the volume of the pure solvent, $V - n_s v_s^0$.
    pub volume_difference: Volume<D>,
    /// Without reactions: residual isobaric heat capacity of the solution minus that of the pure
    /// solvent. With reactions: $\mathrm{d}Q/\mathrm{d}T$ at constant pressure and stoichiometric
    /// amounts (equilibrium heat capacity), $Q = H^\mathrm{res} - n_s h_s^{\mathrm{res},0} +
    /// \sum_r\xi_r\left(\Delta H_r^\circ - \sum_i\nu_{ri}\bar h_i^{\mathrm{res},\infty}\right)$,
    /// evaluated as a central difference with a step of 0.05 K; the heat capacity of the solution
    /// minus that of the pure solvent is this quantity plus the ideal-gas heat capacities of the
    /// given (stoichiometric) amounts of the solutes.
    pub residual_heat_capacity_difference: Entropy<D>,
    /// Residual enthalpy of the solution minus that of the pure solvent.
    pub residual_enthalpy_difference: Energy<D>,
    /// Relative enthalpy: enthalpy of the solution minus that of the given amounts at the reference
    /// states (solutes at infinite dilution, solvent pure),
    /// $L = H^\mathrm{res} - \sum_k n_k\bar h_k^{\mathrm{res,ref}} + \sum_r\xi_r\Delta H_r^\circ$.
    pub relative_enthalpy: Energy<D>,
}

/// Equilibrium state of the solution and the pure solvent.
struct Equilibrium<E, D: DualNum<Primitive = f64> + Copy> {
    solution: State<E, Dyn, D>,
    reference: State<E, Dyn, D>,
    moles: DVector<D>,
    extents: DVector<D>,
}

const MAX_ITER: usize = 100;
// convergence: |F| below TOL or a full (not shortened) Newton step below STEP_TOL times the largest extent; ln phi
// is accurate to about 1e-11 (convergence tolerance of the density iteration)
const TOL: f64 = 1e-10;
const STEP_TOL: f64 = 1e-12;
const DT_HEAT_CAPACITY: f64 = 0.05;

impl<D: DualNum<Primitive = f64, Inner = f64> + Copy> ElectrolyteProperties<D> {
    /// Properties of the solution with the amounts `moles` and the solvent `solvent` at
    /// `temperature` and `pressure`.
    ///
    /// The reference state of the solutes is infinite dilution in the pure solvent: the pure
    /// solvent is evaluated with all other amounts exactly zero.
    pub fn new<E: Residual<Dyn, D>>(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        moles: &Moles<DVector<f64>>,
        solvent: usize,
    ) -> FeosResult<Self> {
        Self::with_reactions(eos, temperature, pressure, moles, solvent, 0.0, &[])
    }

    /// Properties of the solution in chemical equilibrium for the `reactions`, starting from the
    /// (stoichiometric) amounts `moles`; `solvent_molar_mass` in kg/mol defines the molality scale
    /// of the equilibrium constants. The solvent must not take part in the reactions.
    pub fn with_reactions<E: Residual<Dyn, D>>(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        moles: &Moles<DVector<f64>>,
        solvent: usize,
        solvent_molar_mass: f64,
        reactions: &[ReactionEquilibrium],
    ) -> FeosResult<Self> {
        let n0: DVector<f64> = moles.clone().convert_into(MOL);
        for r in reactions {
            if r.stoichiometry.len() != n0.len() {
                return Err(FeosError::IncompatibleComponents(
                    r.stoichiometry.len(),
                    n0.len(),
                ));
            }
            if r.stoichiometry[solvent] != 0.0 {
                return Err(FeosError::NotConverged(
                    "reaction equilibrium: the solvent must not take part in a reaction".into(),
                ));
            }
        }
        let t = temperature.convert_into(KELVIN);
        let p = pressure.convert_into(PASCAL);
        let eq = equilibrium(eos, t, p, &n0, solvent, solvent_molar_mass, reactions)?;
        let (solution, reference) = (&eq.solution, &eq.reference);

        let n_total = eq.moles.sum();
        let x_s = eq.moles[solvent] / n_total;
        let ln_gamma = ln_phi_at(solution, p) - ln_phi_at(reference, p);
        let ln_solvent_activity = ln_gamma[solvent] + x_s.ln();
        let residual_enthalpy_difference =
            solution.residual_enthalpy()? - reference.residual_enthalpy()?;
        let h_ref = reference_enthalpies(reference, t);
        let rgas = RGAS.convert_into(JOULE / (MOL * KELVIN));
        let mut relative = solution.residual_enthalpy()?.convert_into(JOULE) - eq.moles.dot(&h_ref);
        for (xi, r) in eq.extents.iter().zip(reactions) {
            relative += *xi * (rgas * r.reaction_enthalpy(t));
        }

        let residual_heat_capacity_difference = if reactions.is_empty() {
            let cp = |s: &State<E, Dyn, D>| -> FeosResult<Entropy<D>> {
                Ok(s.residual_molar_isobaric_heat_capacity() * s.total_moles()?)
            };
            cp(solution)? - cp(reference)?
        } else {
            let q = |t: f64| -> FeosResult<D> {
                let e = equilibrium(eos, t, p, &n0, solvent, solvent_molar_mass, reactions)?;
                let h_ref = reference_enthalpies(&e.reference, t);
                let mut q = (e.solution.residual_enthalpy()? - e.reference.residual_enthalpy()?)
                    .convert_into(JOULE);
                for (xi, r) in e.extents.iter().zip(reactions) {
                    let nu_h: D = r
                        .stoichiometry
                        .iter()
                        .zip(h_ref.iter())
                        .map(|(nu, h)| *h * *nu)
                        .sum();
                    q += *xi * (D::from(rgas * r.reaction_enthalpy(t)) - nu_h);
                }
                Ok(q)
            };
            let h = DT_HEAT_CAPACITY;
            let dq = (q(t + h)? - q(t - h)?) / (2.0 * h);
            Entropy::new(dq)
        };

        Ok(Self {
            ln_solvent_activity,
            ln_gamma,
            solvent_molefrac: x_s.re(),
            moles: eq.moles.clone(),
            extents: eq.extents.clone(),
            volume_difference: solution.volume()? - reference.volume()?,
            residual_heat_capacity_difference,
            residual_enthalpy_difference,
            relative_enthalpy: Energy::new(relative),
        })
    }
}

/// Logarithm of the fugacity coefficients of `state` referred to the pressure `p` (Pa):
/// $\ln\varphi_i + \ln(p_\mathrm{state}/p)$, i.e. $\ln(f_i/(x_i p))$ with the fugacity of the converged state.
/// The density iteration converges to an absolute pressure tolerance, which at low pressures (e.g. vapor
/// pressures of solutions) is a relative deviation of $p_\mathrm{state}$ from $p$ that enters $\ln\varphi_i$
/// directly; the fugacity of the state does not contain it.
fn ln_phi_at<E: Residual<Dyn, D>, D: DualNum<Primitive = f64, Inner = f64> + Copy>(
    state: &State<E, Dyn, D>,
    p: f64,
) -> DVector<D> {
    let correction = (state.pressure(Contributions::Total).convert_into(PASCAL) / p).ln();
    state.ln_phi().map(|x| x + correction)
}

/// Residual partial molar enthalpies (J/mol) in the reference state: pure solvent and solutes at
/// infinite dilution, $\bar h_k^\mathrm{res} = -RT^2(\partial\ln\varphi_k/\partial T)_{p,n}$.
fn reference_enthalpies<E: Residual<Dyn, D>, D: DualNum<Primitive = f64, Inner = f64> + Copy>(
    reference: &State<E, Dyn, D>,
    t: f64,
) -> DVector<D> {
    let rgas = RGAS.convert_into(JOULE / (MOL * KELVIN));
    let dln_phi_dt: DVector<D> =
        (reference.dln_phi_dt() * Temperature::<D>::from_inner(&KELVIN)).into_value();
    dln_phi_dt.map(|d| d * (-rgas * t * t))
}

/// Density relative to the estimate of [liquid] from which a root is accepted without further starts.
const MIN_LIQUID_DENSITY_RATIO: f64 = 0.25;
/// Density relative to the estimate of [liquid] below which a root is not a liquid and is not used.
const MIN_ROOT_DENSITY_RATIO: f64 = 0.1;

/// Liquid state at the amounts `moles` (mol). The density iteration is started from the default
/// liquid density and then from the density of the pure solvent `rho_solvent` (reduced) scaled by
/// the ratio of the total amount to the amount of solvent, rho_est, and from decreasing fractions of
/// it (for concentrated solutions of large ions the liquid density can be about 0.3 of rho_est).
/// Only mechanically stable roots (dp/drho > 0) with a density of at least MIN_ROOT_DENSITY_RATIO rho_est
/// are used (expanded, gas-like roots of the isotherm are not liquids). The first root with a density of
/// at least MIN_LIQUID_DENSITY_RATIO rho_est is returned; otherwise all starts are used and the densest
/// root is returned, so that where the isotherm has several roots the liquid root is selected
/// independently of which start reaches it. dp/drho is evaluated with the real-valued model and does
/// not use the derivatives cached in the state. Error if no start gives such a root.
fn liquid<E: Residual<Dyn, D>, D: DualNum<Primitive = f64, Inner = f64> + Copy>(
    eos: &E,
    t: f64,
    p: f64,
    moles: &DVector<D>,
    solvent: usize,
    rho_solvent: f64,
) -> FeosResult<State<E, Dyn, D>> {
    let temperature = Temperature::from_inner(&(t * KELVIN));
    let pressure = Pressure::from_inner(&(p * PASCAL));
    let n_total = moles.sum();
    let x = moles / n_total;
    let x_re = x.map(|v| v.re());
    let t_reduced = (t * KELVIN).into_reduced();
    let eos_re = eos.re();
    let composition = || (x.clone(), Moles::new(n_total));
    let rho = rho_solvent / x_re[solvent];
    let starts = std::iter::once(Liquid).chain(
        [1.0, 0.75, 0.5, 0.35, 0.25].map(|f| InitialDensity(Density::from_reduced(f * rho))),
    );
    let mut densest: Option<(f64, State<E, Dyn, D>)> = None;
    let mut error = None;
    for start in starts {
        match State::new_npt(eos, temperature, pressure, composition(), Some(start)) {
            Ok(state) => {
                let density = state.density.into_reduced().re();
                let (_, _, dp_drho) = eos_re.p_dpdrho(t_reduced, density, &x_re);
                if dp_drho <= 0.0 || density < MIN_ROOT_DENSITY_RATIO * rho {
                    error = Some(FeosError::IterationFailed(format!(
                        "liquid state: root of density {density} (reduced, estimate {rho}) with dp/drho = \
                         {dp_drho}"
                    )));
                    continue;
                }
                if density >= MIN_LIQUID_DENSITY_RATIO * rho {
                    return Ok(state);
                }
                if densest.as_ref().is_none_or(|(d, _)| density > *d) {
                    densest = Some((density, state));
                }
            }
            Err(e) => error = Some(e),
        }
    }
    match densest {
        Some((_, state)) => Ok(state),
        None => Err(error.unwrap()),
    }
}

/// Chemical equilibrium of the solution: extents xi of the reactions with
/// F_r = sum_i nu_ri (ln m_i + ln gamma_i^{*,m}) - ln K_r = 0 (solutes; the solvent does not
/// react). With ln m_i + ln gamma_i^{*,m} = ln n_i + ln phi_i - ln N + const, the Jacobian is
/// J_rq = sum_ij nu_ri nu_qj (delta_ij / n_i + (N d ln phi_i / d n_j - 1) / N). The extents are
/// solved with f64 (first for the ideal solution, then with the equation of state); a final
/// Newton step with the generic number type D gives their derivatives (implicit function theorem).
fn equilibrium<E: Residual<Dyn, D>, D: DualNum<Primitive = f64, Inner = f64> + Copy>(
    eos: &E,
    t: f64,
    p: f64,
    n0: &DVector<f64>,
    solvent: usize,
    solvent_molar_mass: f64,
    reactions: &[ReactionEquilibrium],
) -> FeosResult<Equilibrium<E, D>> {
    let nc = n0.len();
    let x0 = DVector::from_fn(nc, |i, _| if i == solvent { 1.0 } else { 0.0 });
    let reference = State::new_npt(
        eos,
        Temperature::from_inner(&(t * KELVIN)),
        Pressure::from_inner(&(p * PASCAL)),
        (x0.map(D::from), Moles::from_inner(&(n0[solvent] * MOL))),
        Some(Liquid),
    )?;
    let rho_solvent = reference.density.into_reduced().re();
    let nr = reactions.len();
    if nr == 0 {
        let moles = n0.map(D::from);
        let solution = liquid(eos, t, p, &moles, solvent, rho_solvent)?;
        return Ok(Equilibrium {
            solution,
            reference,
            moles,
            extents: DVector::zeros(0),
        });
    }
    let nu = DMatrix::from_fn(nr, nc, |r, i| reactions[r].stoichiometry[i]);
    let ln_k = DVector::from_fn(nr, |r, _| reactions[r].ln_k(t));
    let ln_phi_ref = ln_phi_at(&reference, p).map(|x| x.re());
    // ln m_i = ln n_i - ln(n_s M_s), m† = 1 mol/kg
    let ln_ms = (n0[solvent] * solvent_molar_mass).ln();
    let amounts = |xi: &DVector<f64>| n0 + nu.transpose() * xi;
    let reacting: Vec<usize> = (0..nc)
        .filter(|&i| (0..nr).any(|r| nu[(r, i)] != 0.0))
        .collect();

    // strictly positive start: every reaction at the middle of its feasible interval
    let mut xi = DVector::zeros(nr);
    for r in 0..nr {
        let n = amounts(&xi);
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for &i in &reacting {
            let v = nu[(r, i)];
            if v > 0.0 {
                lo = lo.max(-n[i] / v);
            } else if v < 0.0 {
                hi = hi.min(-n[i] / v);
            }
        }
        if !(lo.is_finite() && hi.is_finite() && hi > lo) {
            return Err(FeosError::NotConverged(
                "reaction equilibrium: no feasible extent".into(),
            ));
        }
        xi[r] = 0.5 * (lo + hi);
    }

    // residual and Jacobian; ln_phi_ex = ln phi - ln phi_ref and n_dln_phi_dnj of the solution
    // (zero for the ideal solution)
    let system = |xi: &DVector<f64>, ln_phi_ex: &DVector<f64>, m: &DMatrix<f64>| {
        let n = amounts(xi);
        let n_total = n.sum();
        let mut f = -ln_k.clone();
        let mut jac = DMatrix::zeros(nr, nr);
        for r in 0..nr {
            for &i in &reacting {
                let g = n[i].ln() - ln_ms + ln_phi_ex[i] + (n[solvent] / n_total).ln();
                f[r] += nu[(r, i)] * g;
                for q in 0..nr {
                    for &j in &reacting {
                        let d = if i == j { 1.0 / n[i] } else { 0.0 } + (m[(i, j)] - 1.0) / n_total;
                        jac[(r, q)] += nu[(r, i)] * nu[(q, j)] * d;
                    }
                }
            }
        }
        (f, jac)
    };
    // Newton step, shortened to 0.9 of the distance to a zero amount; returns (xi, shortened)
    let step = |xi: &DVector<f64>,
                f: &DVector<f64>,
                jac: &DMatrix<f64>|
     -> FeosResult<(DVector<f64>, bool)> {
        let dxi = jac.clone().lu().solve(&(-f)).ok_or_else(|| {
            FeosError::NotConverged("reaction equilibrium: singular Jacobian".into())
        })?;
        let n = amounts(xi);
        let dn = nu.transpose() * &dxi;
        let mut alpha: f64 = 1.0;
        for &i in &reacting {
            if dn[i] < 0.0 {
                alpha = alpha.min(-0.9 * n[i] / dn[i]);
            }
        }
        Ok((xi + dxi * alpha, alpha < 1.0))
    };

    let zeros_v = DVector::zeros(nc);
    let zeros_m = DMatrix::zeros(nc, nc);
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let (f, jac) = system(&xi, &zeros_v, &zeros_m);
        if f.amax() < TOL {
            converged = true;
            break;
        }
        let (xi_new, shortened) = step(&xi, &f, &jac)?;
        let small = !shortened && (&xi_new - &xi).amax() <= STEP_TOL * xi.amax().max(1e-300);
        xi = xi_new;
        if small {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(FeosError::NotConverged(
            "reaction equilibrium (ideal)".into(),
        ));
    }
    let state_at =
        |xi: &DVector<f64>| liquid(eos, t, p, &amounts(xi).map(D::from), solvent, rho_solvent);
    let mut converged = false;
    let mut last = None;
    for _ in 0..MAX_ITER {
        let s = state_at(&xi)?;
        let ln_phi_ex = ln_phi_at(&s, p).map(|x| x.re()) - &ln_phi_ref;
        let m = s.n_dln_phi_dnj().map(|x| x.re());
        let (f, jac) = system(&xi, &ln_phi_ex, &m);
        if f.amax() < TOL {
            converged = true;
            last = Some((s, jac));
            break;
        }
        let (xi_new, shortened) = step(&xi, &f, &jac)?;
        let small = !shortened && (&xi_new - &xi).amax() <= STEP_TOL * xi.amax().max(1e-300);
        xi = xi_new;
        if small {
            let s = state_at(&xi)?;
            let ln_phi_ex = ln_phi_at(&s, p).map(|x| x.re()) - &ln_phi_ref;
            let m = s.n_dln_phi_dnj().map(|x| x.re());
            let (_, jac) = system(&xi, &ln_phi_ex, &m);
            converged = true;
            last = Some((s, jac));
            break;
        }
    }
    let (state, jac) = match (converged, last) {
        (true, Some(l)) => l,
        _ => return Err(FeosError::NotConverged("reaction equilibrium".into())),
    };

    // derivatives: xi_D = xi - J^-1 F_D(xi) with F_D of the generic number type at fixed amounts
    let n = amounts(&xi);
    let n_total = n.sum();
    let ln_phi_ex: DVector<D> = ln_phi_at(&state, p) - ln_phi_at(&reference, p);
    let mut f_d: DVector<D> = DVector::from_element(nr, D::zero());
    for r in 0..nr {
        f_d[r] = D::from(-ln_k[r]);
        for &i in &reacting {
            let g = ln_phi_ex[i] + (n[i].ln() - ln_ms + (n[solvent] / n_total).ln());
            f_d[r] += g * nu[(r, i)];
        }
    }
    let jac_inv = jac
        .try_inverse()
        .ok_or_else(|| FeosError::NotConverged("reaction equilibrium: singular Jacobian".into()))?;
    let extents = DVector::from_fn(nr, |r, _| {
        let mut x = D::from(xi[r]);
        for q in 0..nr {
            x -= f_d[q] * jac_inv[(r, q)];
        }
        x
    });
    let moles = DVector::from_fn(nc, |i, _| {
        let mut x = D::from(n0[i]);
        for r in 0..nr {
            x += extents[r] * nu[(r, i)];
        }
        x
    });
    let solution = liquid(eos, t, p, &moles, solvent, rho_solvent)?;
    Ok(Equilibrium {
        solution,
        reference,
        moles,
        extents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubic::{PengRobinson, PengRobinsonParameters};
    use nalgebra::dvector;

    /// Below the critical temperature the liquid root is returned; above it the only root is gas-like and
    /// no state is returned.
    #[test]
    fn liquid_root_selection() -> FeosResult<()> {
        let parameters =
            PengRobinsonParameters::new_simple(&[647.096], &[22.064e6], &[0.3443], &[18.015268])?;
        let eos = &PengRobinson::new(parameters);
        let moles = dvector![55.5];
        let state = liquid(&eos, 300.0, 1e5, &moles, 0, 0.0)?;
        let rho = state.density.into_reduced();
        assert!(state.dp_drho(Contributions::Total).into_reduced() > 0.0);
        let liquid_state = liquid(&eos, 300.0, 1e5, &moles, 0, rho)?;
        assert_eq!(liquid_state.density.into_reduced(), rho);
        assert!(liquid(&eos, 700.0, 1e5, &moles, 0, rho).is_err());
        // vapor root and (metastable) liquid root below the vapor pressure: with an estimate five times the
        // liquid density no root is accepted early, all starts are used and the denser root is returned
        let reference = liquid(&eos, 373.15, 5e4, &moles, 0, rho)?
            .density
            .into_reduced();
        let vapor = State::new_npt(
            &eos,
            Temperature::from_inner(&(373.15 * KELVIN)),
            Pressure::from_inner(&(5e4 * PASCAL)),
            (dvector![1.0], Moles::new(55.5)),
            Some(crate::DensityInitialization::Vapor),
        )?
        .density
        .into_reduced();
        assert!(vapor < 1e-2 * reference);
        let densest = liquid(&eos, 373.15, 5e4, &moles, 0, 5.0 * reference)?;
        assert_eq!(densest.density.into_reduced(), reference);
        Ok(())
    }
}
