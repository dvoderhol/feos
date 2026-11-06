//! ePCP-SAFT with model parameters as dual numbers (derivatives with respect to model parameters).
use super::EPcpSaft;
use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::ad::ElectrolyteProperties;
use feos_core::{FeosError, FeosResult, Residual, StateHD};
use nalgebra::{Const, DVector, Dyn, U1};
use num_dual::{Derivative, DualNum, DualSVec};
use quantity::{JOULE, KELVIN, METER, Moles, Pressure, Temperature};
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

/// Adjustable parameters of ePCP-SAFT (component indices as in the parameter set).
///
/// The association parameters (kappa_ab, epsilon_k_ab) are not included: cross-association
/// parameters are either combined from pure-component values or given in binary records,
/// which cannot be distinguished after the parameters are built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EPcpSaftParameter {
    M(usize),
    Sigma(usize),
    EpsilonK(usize),
    Mu(usize),
    Q(usize),
    KIJ(usize, usize),
    DipoleScaling(usize),
    PolarizabilityScaling(usize),
    CorrelationIntegralParameter(usize),
    BornDeltaCation,
    BornDeltaAnion,
}

/// ePCP-SAFT with all adjustable parameters of the type `D`.
///
/// `EPcpSaftAD<f64>` is identical to [EPcpSaft]; [EPcpSaftAD::derivatives] seeds the
/// derivatives with respect to selected parameters.
#[derive(Clone)]
pub struct EPcpSaftAD<D> {
    model: Arc<EPcpSaft>,
    pub params: EPcpSaftPars<D>,
    /// hard-chain term evaluated (m != 1 for any component, or derivatives with respect to m)
    hard_chain: bool,
}

impl EPcpSaftAD<f64> {
    pub fn new(model: Arc<EPcpSaft>) -> Self {
        let params = model.params.clone();
        let hard_chain = model.hard_chain;
        Self {
            model,
            params,
            hard_chain,
        }
    }

    /// Model with derivatives with respect to `parameters` (gradient component k
    /// corresponds to `parameters[k]`).
    pub fn derivatives<const P: usize>(
        &self,
        parameters: [EPcpSaftParameter; P],
    ) -> FeosResult<EPcpSaftAD<DualSVec<f64, f64, P>>> {
        self.derivatives_partial(&parameters)
    }

    /// Model with derivatives with respect to at most `P` parameters (gradient component k
    /// corresponds to `parameters[k]`; components `parameters.len()..P` are zero).
    pub fn derivatives_partial<const P: usize>(
        &self,
        parameters: &[EPcpSaftParameter],
    ) -> FeosResult<EPcpSaftAD<DualSVec<f64, f64, P>>> {
        use EPcpSaftParameter::*;
        if parameters.len() > P {
            return Err(FeosError::IncompatibleParameters(format!(
                "{} parameters, but only {P} derivative components",
                parameters.len()
            )));
        }
        let p0 = &self.params;
        let n = p0.m.len();
        let mut p = p0.map(DualSVec::from);
        let error = |msg: String| Err(FeosError::IncompatibleParameters(msg));
        check_duplicates(parameters)?;
        for (k, &parameter) in parameters.iter().enumerate() {
            let seed = |x: &mut DualSVec<f64, f64, P>| {
                x.eps = Derivative::derivative_generic(Const::<P>, U1, k)
            };
            let component = match parameter {
                M(i)
                | Sigma(i)
                | EpsilonK(i)
                | Mu(i)
                | Q(i)
                | DipoleScaling(i)
                | PolarizabilityScaling(i)
                | CorrelationIntegralParameter(i) => Some(i),
                KIJ(i, j) => Some(i.max(j)),
                BornDeltaCation | BornDeltaAnion => None,
            };
            if component.is_some_and(|i| i >= n) {
                return error(format!("{parameter:?}: component index out of range"));
            }
            match parameter {
                M(i) => seed(&mut p.m[i]),
                Sigma(i) => seed(&mut p.sigma[i]),
                EpsilonK(i) => {
                    if p0.epsilon_k[i] == 0.0 {
                        return error(format!(
                            "{parameter:?}: epsilon_k = 0, sqrt(epsilon_k_i epsilon_k_j) is not differentiable"
                        ));
                    }
                    seed(&mut p.epsilon_k[i])
                }
                Mu(i) => {
                    if !p0.dipole_comp.contains(&i) {
                        return error(format!(
                            "{parameter:?}: mu = 0, the dipole term does not include this component"
                        ));
                    }
                    seed(&mut p.mu[i])
                }
                Q(i) => {
                    if !p0.quadpole_comp.contains(&i) {
                        return error(format!(
                            "{parameter:?}: q = 0, the quadrupole term does not include this component"
                        ));
                    }
                    seed(&mut p.q[i])
                }
                KIJ(i, j) => {
                    if i == j || p0.z[i] * p0.z[j] > 0.0 {
                        return error(format!(
                            "{parameter:?}: k_ij is fixed (i = j or ions of the same charge sign)"
                        ));
                    }
                    seed(&mut p.k_ij[(i, j)]);
                    seed(&mut p.k_ij[(j, i)]);
                }
                DipoleScaling(i) | PolarizabilityScaling(i) | CorrelationIntegralParameter(i) => {
                    let Some(perm) = p.permittivity.as_mut() else {
                        return error(format!("{parameter:?}: no permittivity records"));
                    };
                    let index = match parameter {
                        DipoleScaling(_) => 0,
                        PolarizabilityScaling(_) => 1,
                        _ => 2,
                    };
                    seed(&mut perm[index][i])
                }
                BornDeltaCation => seed(&mut p.born_delta[0]),
                BornDeltaAnion => seed(&mut p.born_delta[1]),
            }
        }
        let hard_chain = self.hard_chain || parameters.iter().any(|p| matches!(p, M(_)));
        Ok(EPcpSaftAD {
            model: self.model.clone(),
            params: p.map(|x| x),
            hard_chain,
        })
    }
}

impl<D: DualNum<f64> + Copy> Residual<Dyn, D> for EPcpSaftAD<D> {
    fn components(&self) -> usize {
        self.params.m.len()
    }

    type Real = EPcpSaftAD<f64>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = EPcpSaftAD<D2>;

    fn re(&self) -> Self::Real {
        EPcpSaftAD {
            model: self.model.clone(),
            params: self.params.map(|x| x.re()),
            hard_chain: self.hard_chain,
        }
    }

    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        EPcpSaftAD {
            model: self.model.clone(),
            params: self.params.map(|x| D2::from_inner(&x)),
            hard_chain: self.hard_chain,
        }
    }

    fn compute_max_density(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let msigma3 = p.m.component_mul(&p.sigma.map(|v| v.powi(3)));
        (msigma3.dot(molefracs) * FRAC_PI_6).recip() * self.model.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        self.model
            .contributions(&self.params, state, self.hard_chain)
    }
}

/// Number of derivatives evaluated simultaneously by [electrolyte_properties_derivatives].
pub const ELECTROLYTE_GRADIENT_CHUNK: usize = 8;

/// Error if a parameter is requested more than once (k_ij and k_ji are the same parameter).
fn check_duplicates(parameters: &[EPcpSaftParameter]) -> FeosResult<()> {
    use EPcpSaftParameter::KIJ;
    for (k, &parameter) in parameters.iter().enumerate() {
        let same = |&q: &EPcpSaftParameter| match (q, parameter) {
            (KIJ(a, b), KIJ(c, d)) => (a, b) == (c, d) || (a, b) == (d, c),
            _ => q == parameter,
        };
        if parameters[..k].iter().any(same) {
            return Err(FeosError::IncompatibleParameters(format!(
                "{parameter:?}: requested more than once"
            )));
        }
    }
    Ok(())
}

impl EPcpSaftAD<f64> {
    /// Error if the derivatives with respect to `parameters` cannot be calculated (unknown
    /// component, fixed parameter, duplicates; any number of parameters).
    pub fn check_derivatives(&self, parameters: &[EPcpSaftParameter]) -> FeosResult<()> {
        check_duplicates(parameters)?;
        for chunk in parameters.chunks(ELECTROLYTE_GRADIENT_CHUNK) {
            self.derivatives_partial::<ELECTROLYTE_GRADIENT_CHUNK>(chunk)?;
        }
        Ok(())
    }
}

/// A value and its derivatives with respect to model parameters.
#[derive(Clone, Debug)]
pub struct ValueGradient {
    pub value: f64,
    pub gradient: DVector<f64>,
}

/// [feos_core::ad::ElectrolyteProperties] (SI units: m³, J/K, J) and their derivatives with
/// respect to model parameters.
#[derive(Clone, Debug)]
pub struct ElectrolyteDerivatives {
    pub ln_solvent_activity: ValueGradient,
    pub ln_gamma: Vec<ValueGradient>,
    pub volume_difference: ValueGradient,
    pub residual_heat_capacity_difference: ValueGradient,
    pub residual_enthalpy_difference: ValueGradient,
}

/// Electrolyte properties ([feos_core::ad::ElectrolyteProperties]) at `temperature`, `pressure`
/// and amounts `moles` with the solvent `solvent`, and their derivatives with respect to
/// `parameters` (any number; evaluated in chunks of [ELECTROLYTE_GRADIENT_CHUNK]).
pub fn electrolyte_properties_derivatives(
    model: &Arc<EPcpSaft>,
    parameters: &[EPcpSaftParameter],
    temperature: Temperature,
    pressure: Pressure,
    moles: &Moles<DVector<f64>>,
    solvent: usize,
) -> FeosResult<ElectrolyteDerivatives> {
    const P: usize = ELECTROLYTE_GRADIENT_CHUNK;
    check_duplicates(parameters)?;
    let n = moles.len();
    let np = parameters.len();
    let new = |v: f64| ValueGradient {
        value: v,
        gradient: DVector::zeros(np),
    };
    let ad = EPcpSaftAD::new(model.clone());
    let mut out: Option<ElectrolyteDerivatives> = None;
    let chunks: Vec<_> = if parameters.is_empty() {
        vec![&parameters[..0]]
    } else {
        parameters.chunks(P).collect()
    };
    let mut offset = 0;
    for chunk in chunks {
        let eos = ad.derivatives_partial::<P>(chunk)?;
        let e = ElectrolyteProperties::new(&eos, temperature, pressure, moles, solvent)?;
        let values = [
            e.ln_solvent_activity,
            e.volume_difference.convert_into(METER * METER * METER),
            e.residual_heat_capacity_difference
                .convert_into(JOULE / KELVIN),
            e.residual_enthalpy_difference.convert_into(JOULE),
        ];
        let o = out.get_or_insert_with(|| ElectrolyteDerivatives {
            ln_solvent_activity: new(values[0].re),
            ln_gamma: e.ln_gamma.iter().map(|x| new(x.re)).collect(),
            volume_difference: new(values[1].re),
            residual_heat_capacity_difference: new(values[2].re),
            residual_enthalpy_difference: new(values[3].re),
        });
        let set = |target: &mut ValueGradient, x: &DualSVec<f64, f64, P>| {
            let eps = x.eps.unwrap_generic(Const::<P>, U1);
            for k in 0..chunk.len() {
                target.gradient[offset + k] = eps[k];
            }
        };
        set(&mut o.ln_solvent_activity, &values[0]);
        set(&mut o.volume_difference, &values[1]);
        set(&mut o.residual_heat_capacity_difference, &values[2]);
        set(&mut o.residual_enthalpy_difference, &values[3]);
        for i in 0..n {
            set(&mut o.ln_gamma[i], &e.ln_gamma[i]);
        }
        offset += chunk.len();
    }
    Ok(out.unwrap())
}

#[cfg(test)]
mod tests {
    use super::EPcpSaftParameter::*;
    use super::*;
    use crate::epcpsaft::parameters::utils::water_nacl_records;
    use crate::epcpsaft::{EPcpSaftBinaryRecord, EPcpSaftParameters};
    use approx::assert_relative_eq;
    use feos_core::parameter::{BinaryRecord, IdentifierOption};
    use feos_core::{DensityInitialization, ReferenceSystem, ResidualDyn, State};
    use nalgebra::dvector;
    use num_dual::DualStruct;
    use quantity::{BAR, JOULE, KELVIN, METER, MOL, PASCAL, Pressure, Temperature};

    /// water, Na+, Cl- with k_ij(water, Na+) = 0.01, k_ij(Na+, Cl-) = 0.02
    fn water_nacl() -> FeosResult<EPcpSaft> {
        let binary = vec![
            BinaryRecord::new(0, 1, Some(EPcpSaftBinaryRecord::new(0.01))),
            BinaryRecord::new(1, 2, Some(EPcpSaftBinaryRecord::new(0.02))),
        ];
        EPcpSaft::new(EPcpSaftParameters::new(water_nacl_records(), binary)?)
    }

    fn mixture() -> FeosResult<EPcpSaft> {
        EPcpSaft::new(EPcpSaftParameters::from_json(
            vec!["water", "acetone", "methanol"],
            "../../parameters/epcpsaft/esper2023.json",
            Some("../../parameters/epcpsaft/rehner2023_binary.json"),
            IdentifierOption::Name,
        )?)
    }

    /// model with one adjustable parameter changed by `h`
    fn perturbed(eos: &Arc<EPcpSaft>, parameter: EPcpSaftParameter, h: f64) -> EPcpSaftAD<f64> {
        let mut ad = EPcpSaftAD::new(eos.clone());
        let p = &mut ad.params;
        match parameter {
            M(i) => p.m[i] += h,
            Sigma(i) => p.sigma[i] += h,
            EpsilonK(i) => p.epsilon_k[i] += h,
            Mu(i) => p.mu[i] += h,
            Q(i) => p.q[i] += h,
            KIJ(i, j) => {
                p.k_ij[(i, j)] += h;
                p.k_ij[(j, i)] += h
            }
            DipoleScaling(i) => p.permittivity.as_mut().unwrap()[0][i] += h,
            PolarizabilityScaling(i) => p.permittivity.as_mut().unwrap()[1][i] += h,
            CorrelationIntegralParameter(i) => p.permittivity.as_mut().unwrap()[2][i] += h,
            BornDeltaCation => p.born_delta[0] += h,
            BornDeltaAnion => p.born_delta[1] += h,
        }
        ad.params = ad.params.map(|x| x);
        ad
    }

    /// Electrolyte properties (water + NaCl, 1.5 mol/kg) and their derivatives with respect to
    /// 10 parameters (two chunks) agree with central differences (Debye-Hückel and MSA).
    #[test]
    fn electrolyte_properties_gradient() -> FeosResult<()> {
        use super::super::EPcpSaftVariants::*;
        use super::super::MsaPacking::*;
        for (epcpsaft_variant, msa_packing) in [
            (DebyeHueckel, Ions),
            (MeanSphericalApproximation, Ions),
            (MeanSphericalApproximation, Total),
        ] {
            electrolyte_properties_gradient_variant(epcpsaft_variant, msa_packing)?;
        }
        Ok(())
    }

    fn electrolyte_properties_gradient_variant(
        epcpsaft_variant: super::super::EPcpSaftVariants,
        msa_packing: super::super::MsaPacking,
    ) -> FeosResult<()> {
        let options = super::super::EPcpSaftOptions {
            born_delta_cation: 0.1,
            born_delta_anion: -0.05,
            epcpsaft_variant,
            msa_packing,
            ..Default::default()
        };
        let binary = vec![
            BinaryRecord::new(0, 1, Some(EPcpSaftBinaryRecord::new(0.01))),
            BinaryRecord::new(0, 2, Some(EPcpSaftBinaryRecord::new(-0.02))),
        ];
        let eos = Arc::new(EPcpSaft::with_options(
            EPcpSaftParameters::new(water_nacl_records(), binary)?,
            options,
        )?);
        let moles = dvector![1.0 / 0.018015268, 1.5, 1.5] * MOL;
        let (t, p) = (320.0 * KELVIN, 2e5 * PASCAL);
        let parameters = [
            Sigma(1),
            EpsilonK(2),
            KIJ(0, 1),
            KIJ(0, 2),
            DipoleScaling(1),
            CorrelationIntegralParameter(2),
            BornDeltaCation,
            BornDeltaAnion,
            Sigma(2),
            EpsilonK(1),
        ];
        let flatten = |e: &ElectrolyteDerivatives| {
            let mut v = vec![
                e.ln_solvent_activity.clone(),
                e.volume_difference.clone(),
                e.residual_heat_capacity_difference.clone(),
                e.residual_enthalpy_difference.clone(),
            ];
            v.extend(e.ln_gamma.iter().cloned());
            v
        };
        let d = flatten(&electrolyte_properties_derivatives(
            &eos,
            &parameters,
            t,
            p,
            &moles,
            0,
        )?);
        let values: Vec<f64> = d.iter().map(|x| x.value).collect();
        let f64_values = |ad: &EPcpSaftAD<f64>| -> FeosResult<Vec<f64>> {
            let e = ElectrolyteProperties::<f64>::new(ad, t, p, &moles, 0)?;
            let mut v = vec![
                e.ln_solvent_activity,
                e.volume_difference.convert_into(METER * METER * METER),
                e.residual_heat_capacity_difference
                    .convert_into(JOULE / KELVIN),
                e.residual_enthalpy_difference.convert_into(JOULE),
            ];
            v.extend(e.ln_gamma.iter());
            Ok(v)
        };
        let v0 = f64_values(&EPcpSaftAD::new(eos.clone()))?;
        // the density iteration converges to its tolerance with dual and real numbers alike
        for (a, b) in values.iter().zip(&v0) {
            assert_relative_eq!(a, b, max_relative = 1e-8);
        }
        // 4th-order central differences with a step large compared to the convergence
        // tolerance of the density iteration
        let scale = |parameter| match parameter {
            Sigma(_) => 3.0,
            EpsilonK(_) => 200.0,
            DipoleScaling(_) => 5.0,
            BornDeltaCation | BornDeltaAnion => 10.0,
            _ => 1.0,
        };
        for (k, &parameter) in parameters.iter().enumerate() {
            let h = 1e-3 * scale(parameter);
            let v = |h| f64_values(&perturbed(&eos, parameter, h));
            let (v1, v2, v3, v4) = (v(h)?, v(-h)?, v(2.0 * h)?, v(-2.0 * h)?);
            for i in 0..values.len() {
                let fd = (8.0 * (v1[i] - v2[i]) - (v3[i] - v4[i])) / (12.0 * h);
                println!("{parameter:?} {i}: {} {fd}", d[i].gradient[k]);
                assert_relative_eq!(
                    d[i].gradient[k],
                    fd,
                    max_relative = 1e-5,
                    epsilon = 1e-9 * (1.0 + values[i].abs())
                );
            }
        }
        // osmotic coefficient of the solution: physical range
        let phi = -values[0] / (0.018015268 * 3.0);
        assert!(phi > 0.3 && phi < 2.0, "phi = {phi}");
        Ok(())
    }

    /// the whole parameter list is checked (duplicates in different chunks, fixed parameters)
    #[test]
    fn check_derivatives() -> FeosResult<()> {
        let ad = EPcpSaftAD::new(Arc::new(water_nacl()?));
        let mut long = vec![Sigma(1); 1];
        long.extend([
            Sigma(2),
            EpsilonK(1),
            EpsilonK(2),
            KIJ(0, 1),
            KIJ(0, 2),
            M(0),
            Sigma(0),
        ]);
        assert!(ad.check_derivatives(&long).is_ok());
        long.push(Sigma(1)); // duplicate in the second chunk
        assert!(ad.check_derivatives(&long).is_err());
        assert!(ad.check_derivatives(&[KIJ(1, 1)]).is_err());
        assert!(ad.check_derivatives(&[Mu(1)]).is_err());
        Ok(())
    }

    /// central difference of a / (kT) with respect to one adjustable parameter
    fn finite_difference(
        eos: &EPcpSaft,
        state: &StateHD<f64>,
        parameter: EPcpSaftParameter,
        h: f64,
    ) -> f64 {
        let a = |h: f64| {
            let mut p = eos.params.clone();
            match parameter {
                M(i) => p.m[i] += h,
                Sigma(i) => p.sigma[i] += h,
                EpsilonK(i) => p.epsilon_k[i] += h,
                Mu(i) => p.mu[i] += h,
                Q(i) => p.q[i] += h,
                KIJ(i, j) => {
                    p.k_ij[(i, j)] += h;
                    p.k_ij[(j, i)] += h
                }
                DipoleScaling(i) => p.permittivity.as_mut().unwrap()[0][i] += h,
                PolarizabilityScaling(i) => p.permittivity.as_mut().unwrap()[1][i] += h,
                CorrelationIntegralParameter(i) => p.permittivity.as_mut().unwrap()[2][i] += h,
                BornDeltaCation => p.born_delta[0] += h,
                BornDeltaAnion => p.born_delta[1] += h,
            }
            let p = p.map(|x| x);
            eos.contributions(&p, state, true)
                .iter()
                .map(|(_, a)| a)
                .sum::<f64>()
        };
        (a(h) - a(-h)) / (2.0 * h)
    }

    fn check_gradient<const P: usize>(
        eos: EPcpSaft,
        state: StateHD<f64>,
        parameters: [EPcpSaftParameter; P],
    ) -> FeosResult<()> {
        let eos = Arc::new(eos);
        let ad = EPcpSaftAD::new(eos.clone()).derivatives(parameters)?;
        let s = StateHD::new_density(
            DualSVec::from(state.temperature),
            &state.partial_density.map(DualSVec::from),
        );
        let a = ad.reduced_residual_helmholtz_energy_density(&s);
        let a_f64 = ResidualDyn::reduced_helmholtz_energy_density_contributions(&*eos, &state)
            .iter()
            .map(|(_, a)| a)
            .sum::<f64>();
        assert_relative_eq!(a.re, a_f64, max_relative = 1e-14);
        let grad = a.eps.unwrap_generic(Const::<P>, U1);
        for (k, &parameter) in parameters.iter().enumerate() {
            let fd = finite_difference(&eos, &state, parameter, 1e-5);
            println!("{parameter:?}: {} {fd}", grad[k]);
            assert_relative_eq!(grad[k], fd, max_relative = 1e-6, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn same_as_model() -> FeosResult<()> {
        for eos in [water_nacl()?, mixture()?] {
            let n = eos.components();
            let state = StateHD::new(305.0, 30.0, &DVector::from_fn(n, |i, _| [0.8, 0.1, 0.1][i]));
            let eos = Arc::new(eos);
            let ad = EPcpSaftAD::new(eos.clone());
            assert_eq!(
                ad.reduced_helmholtz_energy_density_contributions(&state),
                ResidualDyn::reduced_helmholtz_energy_density_contributions(&*eos, &state)
            );
        }
        Ok(())
    }

    #[test]
    fn gradient_electrolyte() -> FeosResult<()> {
        let state = StateHD::new(310.0, 25.0, &dvector![0.9, 0.05, 0.05]);
        let options = super::super::EPcpSaftOptions {
            born_delta_cation: 0.1,
            born_delta_anion: 0.05,
            ..Default::default()
        };
        let binary = vec![
            BinaryRecord::new(0, 1, Some(EPcpSaftBinaryRecord::new(0.01))),
            BinaryRecord::new(1, 2, Some(EPcpSaftBinaryRecord::new(0.02))),
        ];
        let eos = EPcpSaft::with_options(
            EPcpSaftParameters::new(water_nacl_records(), binary)?,
            options,
        )?;
        check_gradient(
            eos,
            state,
            [
                Sigma(1),
                Sigma(2),
                EpsilonK(1),
                M(0),
                Sigma(0),
                EpsilonK(0),
                KIJ(0, 1),
                KIJ(1, 2),
                DipoleScaling(1),
                PolarizabilityScaling(2),
                CorrelationIntegralParameter(1),
                DipoleScaling(0),
                BornDeltaCation,
                BornDeltaAnion,
            ],
        )
    }

    #[test]
    fn gradient_neutral_polar_cross_association() -> FeosResult<()> {
        let state = StateHD::new(320.0, 60.0, &dvector![0.5, 0.3, 0.2]);
        check_gradient(
            mixture()?,
            state,
            [
                M(0),
                Sigma(0),
                EpsilonK(0),
                Mu(1),
                Sigma(1),
                M(1),
                KIJ(0, 1),
                KIJ(1, 2),
                Sigma(2),
            ],
        )
    }

    #[test]
    fn fixed_parameters() -> FeosResult<()> {
        let ad = EPcpSaftAD::new(Arc::new(water_nacl()?));
        assert!(ad.derivatives([KIJ(1, 1)]).is_err());
        assert!(ad.derivatives([Mu(0)]).is_err());
        assert!(ad.derivatives([Sigma(3)]).is_err());
        assert!(ad.derivatives([Sigma(1), Sigma(1)]).is_err());
        assert!(ad.derivatives([KIJ(0, 1), KIJ(1, 0)]).is_err());
        Ok(())
    }

    /// Ions without dispersion (epsilon_k = 0): finite derivatives with respect to all other
    /// parameters; epsilon_k of these ions cannot be differentiated.
    #[test]
    fn gradient_zero_epsilon_k() -> FeosResult<()> {
        let mut records = water_nacl_records();
        records[1].model_record.epsilon_k = 0.0;
        records[2].model_record.epsilon_k = 0.0;
        let binary = vec![BinaryRecord::new(
            0,
            1,
            Some(EPcpSaftBinaryRecord::new(0.01)),
        )];
        let eos = EPcpSaft::new(EPcpSaftParameters::new(records.clone(), binary.clone())?)?;
        assert!(
            EPcpSaftAD::new(Arc::new(eos))
                .derivatives([EpsilonK(1)])
                .is_err()
        );
        let eos = EPcpSaft::new(EPcpSaftParameters::new(records, binary)?)?;
        let state = StateHD::new(310.0, 25.0, &dvector![0.9, 0.05, 0.05]);
        check_gradient(
            eos,
            state,
            [
                Sigma(1),
                Sigma(2),
                EpsilonK(0),
                KIJ(0, 1),
                M(0),
                DipoleScaling(1),
            ],
        )
    }

    /// All m = 1: the hard-chain term vanishes, its derivative with respect to m does not.
    #[test]
    fn gradient_hard_chain_spheres() -> FeosResult<()> {
        let records = serde_json::from_str(
            r#"[
                {"identifier": {"name": "a"}, "molarweight": 16.0, "m": 1.0, "sigma": 3.7, "epsilon_k": 150.0},
                {"identifier": {"name": "b"}, "molarweight": 40.0, "m": 1.0, "sigma": 3.4, "epsilon_k": 120.0}
            ]"#,
        )
        .expect("Unable to parse json.");
        let eos = EPcpSaft::new(EPcpSaftParameters::new(records, vec![])?)?;
        assert!(!eos.hard_chain);
        let state = StateHD::new(200.0, 100.0, &dvector![0.6, 0.4]);
        check_gradient(eos, state, [M(0), M(1), Sigma(0)])
    }

    /// Derivative of the liquid density at given (T, p) (implicit differentiation in feos-core)
    #[test]
    fn density_derivative() -> FeosResult<()> {
        let (t, p, x) = (298.15 * KELVIN, BAR, dvector![0.9, 0.05, 0.05]);
        let rho = |dsigma: f64| -> FeosResult<f64> {
            let mut records = water_nacl_records();
            records[1].model_record.sigma += dsigma;
            let eos = EPcpSaft::new(EPcpSaftParameters::new(records, vec![])?)?;
            let s = State::new_npt(&&eos, t, p, &x, Some(DensityInitialization::Liquid))?;
            Ok(s.density.into_reduced())
        };
        let model = EPcpSaft::new(EPcpSaftParameters::new(water_nacl_records(), vec![])?)?;
        let ad = EPcpSaftAD::new(Arc::new(model)).derivatives([Sigma(1)])?;
        let s = State::new_npt(
            &ad,
            Temperature::from_inner(&t),
            Pressure::from_inner(&p),
            x.map(DualSVec::from),
            Some(DensityInitialization::Liquid),
        )?;
        let h = 1e-6;
        let fd = (rho(h)? - rho(-h)?) / (2.0 * h);
        let d = s.density.into_reduced();
        assert_relative_eq!(d.re, rho(0.0)?, max_relative = 1e-12);
        assert_relative_eq!(
            d.eps.unwrap_generic(Const::<1>, U1)[0],
            fd,
            max_relative = 1e-5
        );
        Ok(())
    }
}
