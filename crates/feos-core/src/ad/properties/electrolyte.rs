use crate::DensityInitialization::{InitialDensity, Liquid};
use crate::{FeosResult, ReferenceSystem, Residual, State};
use nalgebra::{DVector, Dyn};
use num_dual::{DualNum, DualStruct};
use quantity::{Density, Energy, Entropy, Moles, Pressure, Temperature, Volume};

/// Properties of an electrolyte solution relative to the pure solvent at the same temperature
/// and pressure.
///
/// The solution consists of the amounts `moles`, usually per kg of solvent. Differences of
/// extensive properties refer to the pure solvent with the amount `moles[solvent]`, i.e., they
/// are per kg of solvent if `moles` is given per kg of solvent. Residual properties refer to the
/// ideal gas at the same temperature, pressure and composition; ideal-gas contributions of the
/// solutes are not included.
pub struct ElectrolyteProperties<D> {
    /// Natural logarithm of the solvent activity $\ln a_s = \ln x_s + \ln\varphi_s - \ln\varphi_s^0$.
    pub ln_solvent_activity: D,
    /// Natural logarithm of the activity coefficients in the mole-fraction scale: solutes with
    /// the reference state of infinite dilution in the pure solvent,
    /// $\ln\gamma_i^{*} = \ln\varphi_i - \ln\varphi_i^\infty$; solvent with the pure solvent as
    /// reference, $\ln\gamma_s = \ln\varphi_s - \ln\varphi_s^0$. Molality scale:
    /// $\ln\gamma_i^{*,m} = \ln\gamma_i^{*} + \ln x_s$.
    pub ln_gamma: DVector<D>,
    /// Mole fraction of the solvent.
    pub solvent_molefrac: f64,
    /// Volume of the solution minus the volume of the pure solvent, $V - n_s v_s^0$.
    pub volume_difference: Volume<D>,
    /// Residual isobaric heat capacity of the solution minus that of the pure solvent.
    pub residual_heat_capacity_difference: Entropy<D>,
    /// Residual enthalpy of the solution minus that of the pure solvent.
    pub residual_enthalpy_difference: Energy<D>,
}

impl<D: DualNum<f64, Inner = f64> + Copy> ElectrolyteProperties<D> {
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
        let t = Temperature::from_inner(&temperature);
        let p = Pressure::from_inner(&pressure);
        let n_total: Moles = moles.sum();
        let x = (moles / n_total).into_value();
        let x_s = x[solvent];
        let x0 = DVector::from_fn(x.len(), |i, _| if i == solvent { 1.0 } else { 0.0 });
        let n0 = Moles::from_inner(&moles.get(solvent));

        let reference = State::new_npt(eos, t, p, (x0.map(D::from), n0), Some(Liquid))?;
        // At high ion concentrations the density iteration started from the default liquid density
        // can fail although a liquid solution exists; then it is started from the density of the
        // pure solvent scaled by the ratio of the total amount to the amount of solvent, and from
        // decreasing fractions of that density (for concentrated solutions of large ions the liquid
        // density can be about 0.3 of this estimate).
        let composition = || (x.map(D::from), Moles::from_inner(&n_total));
        let mut solution = State::new_npt(eos, t, p, composition(), Some(Liquid));
        let rho = reference.density.into_reduced().re() / x_s;
        for f in [1.0, 0.75, 0.5, 0.35, 0.25] {
            if solution.is_ok() {
                break;
            }
            let init = InitialDensity(Density::from_reduced(f * rho));
            solution = State::new_npt(eos, t, p, composition(), Some(init));
        }
        let solution = solution?;

        let ln_phi = solution.ln_phi();
        let ln_gamma = &ln_phi - &reference.ln_phi();
        let ln_solvent_activity = ln_gamma[solvent] + x_s.ln();

        let cp = |s: &State<E, Dyn, D>| -> FeosResult<Entropy<D>> {
            Ok(s.residual_molar_isobaric_heat_capacity() * s.total_moles()?)
        };
        Ok(Self {
            ln_solvent_activity,
            ln_gamma,
            solvent_molefrac: x_s,
            volume_difference: solution.volume()? - reference.volume()?,
            residual_heat_capacity_difference: cp(&solution)? - cp(&reference)?,
            residual_enthalpy_difference: solution.residual_enthalpy()?
                - reference.residual_enthalpy()?,
        })
    }
}
