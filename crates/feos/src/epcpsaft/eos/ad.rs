//! ePCP-SAFT with model parameters as dual numbers (derivatives with respect to model parameters).
use super::EPcpSaft;
use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::{FeosError, FeosResult, Residual, StateHD};
use nalgebra::{Const, DVector, Dyn, U1};
use num_dual::{Derivative, DualNum, DualSVec};
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
}

impl EPcpSaftAD<f64> {
    pub fn new(model: Arc<EPcpSaft>) -> Self {
        let params = model.params.clone();
        Self { model, params }
    }

    /// Model with derivatives with respect to `parameters` (gradient component k
    /// corresponds to `parameters[k]`).
    pub fn derivatives<const P: usize>(
        &self,
        parameters: [EPcpSaftParameter; P],
    ) -> FeosResult<EPcpSaftAD<DualSVec<f64, f64, P>>> {
        use EPcpSaftParameter::*;
        let p0 = &self.params;
        let n = p0.m.len();
        let mut p = p0.map(DualSVec::from);
        let error = |msg: String| Err(FeosError::IncompatibleParameters(msg));
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
                EpsilonK(i) => seed(&mut p.epsilon_k[i]),
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
        Ok(EPcpSaftAD {
            model: self.model.clone(),
            params: p.map(|x| x),
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
        }
    }

    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        EPcpSaftAD {
            model: self.model.clone(),
            params: self.params.map(|x| D2::from_inner(&x)),
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
        self.model.contributions(&self.params, state)
    }
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
    use quantity::{BAR, KELVIN, Pressure, Temperature};

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
            eos.contributions(&p, state)
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
        Ok(())
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
