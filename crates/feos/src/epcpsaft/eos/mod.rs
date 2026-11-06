use super::parameters::{EPcpSaftAssociationRecord, EPcpSaftParameters, EPcpSaftPars};
use crate::association::AssociationStrength;
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::{
    EntropyScaling, FeosResult, Molarweight, ReferenceSystem, Residual, ResidualDyn, StateHD,
    Subset,
};
use nalgebra::DVector;
use num_dual::{DualNum, partial2};
use quantity::ad::first_derivative;
use quantity::*;
use std::f64::consts::{FRAC_PI_6, PI};

mod ad;
mod association;
mod born;
pub(crate) mod dispersion;
mod epcpsaft_binary;
mod epcpsaft_pure;
pub(crate) mod hard_chain;
mod ionic;
mod msa;
pub(crate) mod permittivity;
pub(crate) mod polar;
pub use ad::{
    ELECTROLYTE_GRADIENT_CHUNK, EPcpSaftAD, EPcpSaftParameter, ElectrolyteDerivatives,
    ValueGradient, electrolyte_properties_derivatives,
};
use association::Association;
use born::Born;
use dispersion::Dispersion;
pub use epcpsaft_binary::EPcpSaftBinary;
pub use epcpsaft_pure::EPcpSaftPure;
use hard_chain::HardChain;
use ionic::Ionic;
use msa::Msa;
pub use msa::MsaPacking;
pub use polar::DQVariants;
use polar::{Dipole, DipoleQuadrupole, Quadrupole};

/// Implemented variants of the ePCP-SAFT equation of state: the term for the electrostatic
/// interactions between ions.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum EPcpSaftVariants {
    /// Debye-Hückel term (as in ePC-SAFT).
    DebyeHueckel,
    /// Mean spherical approximation of the primitive model (Blum 1975).
    MeanSphericalApproximation,
}

/// Customization options for the ePCP-SAFT equation of state and functional.
#[derive(Copy, Clone)]
pub struct EPcpSaftOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
    pub dq_variant: DQVariants,
    /// Term for the electrostatic interactions between ions
    pub epcpsaft_variant: EPcpSaftVariants,
    /// Species in the packing term of the MSA (only for [EPcpSaftVariants::MeanSphericalApproximation])
    pub msa_packing: MsaPacking,
    /// Correction added to the diameter of all cations in the Born term (Angstrom)
    pub born_delta_cation: f64,
    /// Correction added to the diameter of all anions in the Born term (Angstrom)
    pub born_delta_anion: f64,
}

impl Default for EPcpSaftOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            dq_variant: DQVariants::DQ35,
            epcpsaft_variant: EPcpSaftVariants::DebyeHueckel,
            msa_packing: MsaPacking::Ions,
            born_delta_cation: 0.0,
            born_delta_anion: 0.0,
        }
    }
}

/// ePCP-SAFT equation of state: PCP-SAFT with Debye-Hückel and Born contributions
/// and the permittivity model of Rueben et al. (2024).
pub struct EPcpSaft {
    pub parameters: EPcpSaftParameters,
    params: EPcpSaftPars,
    options: EPcpSaftOptions,
    hard_chain: bool,
    dipole: bool,
    quadrupole: bool,
    dipole_quadrupole: bool,
    association: Option<Association>,
    ionic: bool,
}

impl EPcpSaft {
    pub fn new(parameters: EPcpSaftParameters) -> FeosResult<Self> {
        Self::with_options(parameters, EPcpSaftOptions::default())
    }

    pub fn with_options(
        parameters: EPcpSaftParameters,
        options: EPcpSaftOptions,
    ) -> FeosResult<Self> {
        let mut params = EPcpSaftPars::new(&parameters)?;
        params.born_delta = [options.born_delta_cation, options.born_delta_anion];
        let hard_chain = params.m.iter().any(|m| (m - 1.0).abs() > 1e-15);

        let dipole = params.ndipole > 0;
        let quadrupole = params.nquadpole > 0;
        let dipole_quadrupole = params.ndipole > 0 && params.nquadpole > 0;

        let association = (!parameters.association.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));

        let ionic = params.nionic > 0;

        Ok(Self {
            parameters,
            params,
            options,
            hard_chain,
            dipole,
            quadrupole,
            dipole_quadrupole,
            association,
            ionic,
        })
    }

    /// The parameters in an easier accessible format.
    pub fn params(&self) -> &EPcpSaftPars {
        &self.params
    }

    /// Relative permittivity of the mixture; `None` if a component has no permittivity record.
    pub fn permittivity<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> Option<D> {
        self.params.map(D::from).permittivity(state)
    }

    /// Diameters used in the Born term: hard-sphere diameter plus `born_delta_cation`
    /// (cations) or `born_delta_anion` (anions).
    pub fn born_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        self.params.map(D::from).born_diameter(temperature)
    }

    /// Reduced Helmholtz energy densities of all contributions for parameters `p`
    /// (the parameters of this model, or the same parameters as dual numbers).
    /// `hard_chain`: evaluate the hard-chain term (also needed if all m = 1 but derivatives
    /// with respect to m are calculated).
    pub(crate) fn contributions<D: DualNum<f64> + Copy>(
        &self,
        p: &EPcpSaftPars<D>,
        state: &StateHD<D>,
        hard_chain: bool,
    ) -> Vec<(&'static str, D)> {
        let mut v = Vec::with_capacity(9);
        let d = p.diameter(state.temperature);

        v.push(("Hard Sphere", hard_sphere(p, state)));
        if hard_chain {
            v.push(("Hard Chain", HardChain.helmholtz_energy_density(p, state)))
        }
        v.push(("Dispersion", Dispersion.helmholtz_energy_density(p, state)));
        if self.dipole {
            v.push(("Dipole", Dipole.helmholtz_energy_density(p, state)))
        }
        if self.quadrupole {
            v.push(("Quadrupole", Quadrupole.helmholtz_energy_density(p, state)))
        }
        if self.dipole_quadrupole {
            v.push((
                "DipoleQuadrupole",
                DipoleQuadrupole.helmholtz_energy_density(p, state, self.options.dq_variant),
            ))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association",
                association.helmholtz_energy_density(p, &self.parameters.association, state, &d),
            ))
        }
        if self.ionic {
            let permittivity = p.permittivity(state).unwrap();
            let bjerrum_length = p.bjerrum_length(state.temperature, permittivity);
            v.push(match self.options.epcpsaft_variant {
                EPcpSaftVariants::DebyeHueckel => (
                    "Ionic",
                    Ionic.helmholtz_energy_density(p, state, &d, bjerrum_length),
                ),
                EPcpSaftVariants::MeanSphericalApproximation => (
                    "MSA",
                    Msa(self.options.msa_packing).helmholtz_energy_density(
                        p,
                        state,
                        &d,
                        bjerrum_length,
                    ),
                ),
            });
            v.push((
                "Born",
                Born.helmholtz_energy_density(
                    p,
                    state,
                    &p.born_diameter(state.temperature),
                    bjerrum_length,
                    permittivity,
                ),
            ));
        }
        v
    }
}

impl<D: DualNum<f64> + Copy> EPcpSaftPars<D> {
    /// Hard-sphere diameter: PC-SAFT diameter for neutral components,
    /// temperature-independent d_i = (1 - 0.12) sigma_i for ions (Held et al. 2014).
    pub fn diameter(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            if self.z[i] != 0.0 {
                self.sigma[i] * (1.0 - 0.12)
            } else {
                -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
            }
        })
    }

    /// zeta_k = pi / 6 sum_i rho_i m_i d_i^k
    pub fn packing<const N: usize>(
        &self,
        temperature: D,
        partial_density: &DVector<D>,
        k: [i32; N],
    ) -> [D; N] {
        let diameter = self.diameter(temperature);
        let mut zeta = [D::zero(); N];
        for i in 0..diameter.len() {
            for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                *z += partial_density[i] * diameter[i].powi(k) * (self.m[i] * FRAC_PI_6);
            }
        }
        zeta
    }

    /// Born diameter: hard-sphere diameter plus born_delta[0] (cations) or born_delta[1] (anions).
    pub fn born_diameter(&self, temperature: D) -> DVector<D> {
        let mut d = self.diameter(temperature);
        for &i in &self.ionic_comp {
            d[i] += if self.z[i] > 0.0 {
                self.born_delta[0]
            } else {
                self.born_delta[1]
            };
        }
        d
    }
}

/// Hard-sphere contribution (Boublik, Mansoori, Carnahan, Starling, Leland), as in `crate::hard_sphere`.
fn hard_sphere<D: DualNum<f64> + Copy>(p: &EPcpSaftPars<D>, state: &StateHD<D>) -> D {
    let diameter = p.diameter(state.temperature);
    let mut zeta = [D::zero(); 4];
    for i in 0..diameter.len() {
        for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
            *z += state.molefracs[i] * diameter[i].powi(k) * (p.m[i] * FRAC_PI_6);
        }
    }
    let zeta_23 = zeta[2] / zeta[3];
    let density = state.partial_density.sum();
    zeta.iter_mut().for_each(|z| *z *= density);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();
    (zeta[1] * zeta[2] * frac_1mz3 * 3.0
        + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
        + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (-zeta[3]).ln_1p())
        / FRAC_PI_6
}

impl ResidualDyn for EPcpSaft {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        self.contributions(&self.params.map(D::from), state, self.hard_chain)
    }
}

impl Subset for EPcpSaft {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options).unwrap()
    }
}

impl Molarweight for EPcpSaft {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

/// Used by the Helmholtz energy functional.
impl HardSphereProperties for EPcpSaftPars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::NonSpherical(self.m.map(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        self.map(D::from).diameter(temperature)
    }
}

/// Used by the Helmholtz energy functional.
impl AssociationStrength for EPcpSaftPars {
    type Record = EPcpSaftAssociationRecord;

    fn association_strength_ij<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        association_parameters_ij: &Self::Record,
    ) -> D {
        let f_ab = (temperature.recip() * association_parameters_ij.epsilon_k_ab).exp_m1();
        let k_ab = association_parameters_ij.kappa_ab
            * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);
        f_ab * k_ab
    }
}

fn omega11(t: f64) -> f64 {
    1.06036 * t.powf(-0.15610)
        + 0.19300 * (-0.47635 * t).exp()
        + 1.03587 * (-1.52996 * t).exp()
        + 1.76474 * (-3.89411 * t).exp()
}

fn omega22(t: f64) -> f64 {
    1.16145 * t.powf(-0.14874) + 0.52487 * (-0.77320 * t).exp() + 2.16178 * (-2.43787 * t).exp()
        - 6.435e-4 * t.powf(0.14874) * (18.0323 * t.powf(-0.76830) - 7.27371).sin()
}

#[inline]
fn chapman_enskog_thermal_conductivity(
    temperature: Temperature,
    molarweight: MolarWeight,
    m: f64,
    sigma: f64,
    epsilon_k: f64,
) -> ThermalConductivity {
    let t = temperature.into_reduced();
    0.083235 * (t * m / molarweight.convert_to(GRAM / MOL)).sqrt()
        / sigma.powi(2)
        / omega22(t / epsilon_k)
        * WATT
        / METER
        / KELVIN
}

impl EntropyScaling for EPcpSaft {
    fn viscosity_reference(
        &self,
        temperature: Temperature,
        _: MolarVolume,
        molefracs: &DVector<f64>,
    ) -> Viscosity {
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let ce: Vec<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                5.0 / 16.0 * (mw.get(i) * KB / NAV * temperature / PI).sqrt()
                    / omega22(tr)
                    / (p.sigma[i] * ANGSTROM).powi::<2>()
            })
            .collect();
        let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
        for i in 0..self.components() {
            let denom: f64 = (0..self.components())
                .map(|j| {
                    molefracs[j]
                        * (1.0
                            + (ce[i] / ce[j]).into_value().sqrt()
                                * (mw.get(j) / mw.get(i)).powf(1.0 / 4.0))
                        .powi(2)
                        / (8.0 * (1.0 + (mw.get(i) / mw.get(j)).into_value())).sqrt()
                })
                .sum();
            ce_mix += ce[i] * molefracs[i] / denom
        }
        ce_mix
    }

    fn viscosity_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        let coefficients = self
            .params
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        a + b * s + c * s.powi(2) + d * s.powi(3)
    }

    fn diffusion_reference(
        &self,
        temperature: Temperature,
        molar_volume: MolarVolume,
        _: &DVector<f64>,
    ) -> Diffusivity {
        if self.components() != 1 {
            panic!("Diffusion coefficients in PC-SAFT are only implemented for pure components!");
        }
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let res: Vec<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<2>() / omega11(tr)
                    * (molar_volume / NAV)
                    * (temperature * RGAS / PI / mw.get(i) / p.m[i]).sqrt()
            })
            .collect();
        res[0]
    }

    fn diffusion_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        if self.components() != 1 {
            panic!("Diffusion coefficients in PC-SAFT are only implemented for pure components!");
        }
        let coefficients = self
            .params
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        let e = coefficients.row(4).transpose().dot(&pref);
        a + b * s - c * (1.0 - s.exp()) * s.powi(2) - d * s.powi(4) - e * s.powi(8)
    }

    // Equation 4 of DOI: 10.1021/acs.iecr.9b04289
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature,
        molar_volume: MolarVolume,
        molefracs: &DVector<f64>,
    ) -> ThermalConductivity {
        if self.components() != 1 {
            panic!("Thermal conductivity in PC-SAFT is only implemented for pure components!");
        }
        let p = &self.params;
        let mws = self.molar_weight();
        let (_, s_res) = first_derivative(
            partial2(
                |t, &v, n| -self.residual_molar_helmholtz_energy(t, v, n),
                &molar_volume,
                molefracs,
            ),
            temperature,
        );
        let res: Vec<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = s_res.into_reduced() / p.m[i];
                let ref_ce = chapman_enskog_thermal_conductivity(
                    temperature,
                    mws.get(i),
                    p.m[i],
                    p.sigma[i],
                    p.epsilon_k[i],
                );
                let alpha_visc = (-s_res_reduced / -0.5).exp();
                let ref_ts = (-0.0167141 * tr / p.m[i] + 0.0470581 * (tr / p.m[i]).powi(2))
                    * (p.m[i] * p.m[i] * p.sigma[i].powi(3) * p.epsilon_k[i])
                    * 1e-5
                    * WATT
                    / METER
                    / KELVIN;
                ref_ce + ref_ts * alpha_visc
            })
            .collect();
        res[0]
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        if self.components() != 1 {
            panic!("Thermal conductivity in PC-SAFT is only implemented for pure components!");
        }
        let coefficients = self
            .params
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        a + b * s + c * (1.0 - s.exp()) + d * s.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcpsaft::parameters::utils::{
        butane_parameters, nonane_heptane_parameters, nonane_parameters, propane_butane_parameters,
        propane_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::*;
    use nalgebra::dvector;
    use quantity::{BAR, KELVIN, METER, PASCAL, RGAS};

    #[test]
    fn ideal_gas_pressure() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<3>();
        let n = dvector![1.0] * MOL;
        let s = State::new_nvt(&e, t, v, n).unwrap();
        let p_ig = s.total_moles()? * RGAS * t / v;
        assert_relative_eq!(s.pressure(Contributions::IdealGas), p_ig, epsilon = 1e-10);
        assert_relative_eq!(
            s.pressure(Contributions::IdealGas) + s.pressure(Contributions::Residual),
            s.pressure(Contributions::Total),
            epsilon = 1e-10
        );
        Ok(())
    }

    #[test]
    fn ideal_gas_heat_capacity_joback() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<3>();
        let n = dvector![1.0] * MOL;
        let s = State::new_nvt(&e, t, v, n).unwrap();
        let p_ig = s.total_moles()? * RGAS * t / v;
        assert_relative_eq!(s.pressure(Contributions::IdealGas), p_ig, epsilon = 1e-10);
        assert_relative_eq!(
            s.pressure(Contributions::IdealGas) + s.pressure(Contributions::Residual),
            s.pressure(Contributions::Total),
            epsilon = 1e-10
        );
        Ok(())
    }

    #[test]
    fn hard_sphere_pure() {
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a_rust = super::hard_sphere(&propane_parameters().params, &s) * v;
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn hard_sphere_mix() {
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 = super::hard_sphere(&propane_parameters().params, &s);
        let a2 = super::hard_sphere(&butane_parameters().params, &s);
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = super::hard_sphere(&propane_butane_parameters().params, &s1m);
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = super::hard_sphere(&propane_butane_parameters().params, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    /// The hard-sphere term of this module equals `crate::hard_sphere::HardSphere`.
    #[test]
    fn hard_sphere_generic() {
        use crate::hard_sphere::HardSphere;
        let s = StateHD::new(250.0, 100.0, &dvector![0.3, 0.7]);
        let p = &propane_butane_parameters().params;
        assert_eq!(
            super::hard_sphere(p, &s),
            HardSphere.helmholtz_energy_density(p, &s)
        );
    }

    #[test]
    fn new_tpn() {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let m = dvector![1.5] * MOL;
        let s = State::new_npt(&e, t, p, m, None);
        let p_calc = if let Ok(state) = s {
            state.pressure(Contributions::Total)
        } else {
            0.0 * PASCAL
        };
        assert_relative_eq!(p, p_calc, epsilon = 1e-6);
    }

    #[test]
    fn vle_pure() {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let vle = PhaseEquilibrium::pure(&e, t, None, Default::default());
        if let Ok(v) = vle {
            assert_relative_eq!(
                v.vapor().pressure(Contributions::Total),
                v.liquid().pressure(Contributions::Total),
                epsilon = 1e-6
            )
        }
    }

    #[test]
    fn critical_point() {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, (), Some(t), None, Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = &propane_parameters();
        let e2 = &butane_parameters();
        let e12 = &propane_butane_parameters();
        let t = 300.0 * KELVIN;
        let v = 0.02456883872966545 * METER.powi::<3>();
        let m1 = dvector![2.0] * MOL;
        let m1m = dvector![2.0, 0.0] * MOL;
        let m2m = dvector![0.0, 2.0] * MOL;
        let s1 = State::new_nvt(&e1, t, v, &m1).unwrap();
        let s2 = State::new_nvt(&e2, t, v, m1).unwrap();
        let s1m = State::new_nvt(&e12, t, v, m1m).unwrap();
        let s2m = State::new_nvt(&e12, t, v, m2m).unwrap();
        assert_relative_eq!(
            s1.pressure(Contributions::Total),
            s1m.pressure(Contributions::Total),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            s2.pressure(Contributions::Total),
            s2m.pressure(Contributions::Total),
            epsilon = 1e-12
        )
    }

    #[test]
    fn viscosity() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, n, None)?;
        assert_relative_eq!(
            s.viscosity(),
            0.00797 * MILLI * PASCAL * SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_viscosity_reduced(),
            (s.viscosity() / e.viscosity_reference(s.temperature, s.molar_volume, &s.molefracs))
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }

    #[test]
    fn viscosity_mix() -> FeosResult<()> {
        // Test case: compare fig 15 of Lötgering-Lin 2018 (https://doi.org/10.1021/acs.iecr.7b04871)
        let e = &nonane_heptane_parameters();
        let nonane = &nonane_parameters();

        let t = 303.15 * KELVIN;
        let p = 500.0 * BAR;
        let n = dvector![0.25, 0.75] * MOL;
        let viscosity_mix = State::new_npt(&e, t, p, n, None)?.viscosity();
        let viscosity_paper = 0.68298 * MILLI * PASCAL * SECOND;
        assert_relative_eq!(viscosity_paper, viscosity_mix, epsilon = 1e-8);

        // Make sure pure substance case is recovered
        let n_pseudo_mix = dvector![1.0, 0.0] * MOL;
        let viscosity_pseudo_mix = State::new_npt(&e, t, p, n_pseudo_mix, None)?.viscosity();
        let n_nonane = dvector![1.0] * MOL;
        let viscosity_nonane = State::new_npt(&nonane, t, p, n_nonane, None)?.viscosity();
        assert_relative_eq!(viscosity_pseudo_mix, viscosity_nonane, epsilon = 1e-15);
        Ok(())
    }

    #[test]
    fn diffusion() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, n, None)?;
        assert_relative_eq!(
            s.diffusion(),
            0.01505 * (CENTI * METER).powi::<2>() / SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_diffusion_reduced(),
            (s.diffusion() / e.diffusion_reference(s.temperature, s.molar_volume, &s.molefracs))
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests_parameter_fit {
    use super::epcpsaft_binary::test::epcpsaft_binary;
    use super::epcpsaft_pure::test::pcsaft;
    use super::*;
    use approx::assert_relative_eq;
    use feos_core::DensityInitialization::Liquid;
    use feos_core::ad::{
        BoilingTemperature, BubblePointPressure, DewPointPressure, EquilibriumLiquidDensity,
        LiquidDensity, PropertyAD, VaporPressure,
    };
    use feos_core::{Contributions, ReferenceSystem, SolverOptions};
    use feos_core::{FeosResult, PhaseEquilibrium, State, ad::ParametersAD};
    use nalgebra::{U1, U3, U8};
    use num_dual::{Dual64, DualStruct, DualVec, partial};
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL};

    fn flat_binary_params<const N: usize>(b: &EPcpSaftBinary<f64, N>) -> Vec<f64> {
        b.0.0
            .iter()
            .flatten()
            .copied()
            .chain(std::iter::once(b.0.1))
            .collect()
    }

    fn pcsaft_non_assoc() -> EPcpSaftPure<f64, 4> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let params = [m, sigma, epsilon_k, mu];
        EPcpSaftPure(params)
    }

    #[test]
    fn test_vapor_pressure_derivatives() -> FeosResult<()> {
        let pcsaft_params = [
            "m",
            "sigma",
            "epsilon_k",
            "mu",
            "kappa_ab",
            "epsilon_k_ab",
            "na",
            "nb",
        ];
        let (pcsaft, _) = pcsaft()?;
        let pcsaft_ad = EPcpSaftPure::<f64, 8>::seed_derivatives(&pcsaft.0, pcsaft_params);
        let temperature = 250.0 * KELVIN;
        let p = VaporPressure(temperature).evaluate(&pcsaft_ad)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U8, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in pcsaft_params.into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = EPcpSaftPure(params);
            let (p_h, _) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_vapor_pressure_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad =
            EPcpSaftPure::<f64, 4>::seed_derivatives(&pcsaft.0, ["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let p = VaporPressure(temperature).evaluate(&pcsaft_ad)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U3, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = EPcpSaftPure(params);
            let (p_h, _) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_boiling_temperature_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad =
            EPcpSaftPure::<f64, 4>::seed_derivatives(&pcsaft.0, ["m", "sigma", "epsilon_k"]);
        let pressure = BAR;
        let t = BoilingTemperature(pressure).evaluate(&pcsaft_ad)?;
        let t = t.convert_into(KELVIN);
        let (t, grad) = (t.re, t.eps.unwrap_generic(U3, U1));

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (t_check, _) = PhaseEquilibrium::pure_p(
            &pcsaft_ad,
            Pressure::from_inner(&pressure),
            None,
            Default::default(),
        )?;
        let t_check = t_check.convert_into(KELVIN);
        let (t_check, grad_check) = (t_check.re, t_check.eps.unwrap_generic(U3, U1));
        println!("{t_check:.5}");
        println!("{grad_check:.5?}");
        assert_relative_eq!(t, t_check, max_relative = 1e-15);
        assert_relative_eq!(grad, grad_check, max_relative = 1e-15);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-8;
            params[i] += h;
            let pcsaft_h = EPcpSaftPure(params);
            let (t_h, _) = PhaseEquilibrium::pure_p(&pcsaft_h, pressure, None, Default::default())?;
            let dt_h = (t_h.convert_into(KELVIN) - t) / h;
            let dt = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dt_h,
                dt,
                ((dt_h - dt) / dt).abs()
            );
            assert_relative_eq!(dt, dt_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_equilibrium_liquid_density_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad =
            EPcpSaftPure::<f64, 4>::seed_derivatives(&pcsaft.0, ["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let rho = EquilibriumLiquidDensity(temperature).evaluate(&pcsaft_ad)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{rho:.5}");
        println!("{rho_grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = EPcpSaftPure(params);
            let (_, [_, rho_h]) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let drho_h = (rho_h.convert_into(MOL / LITER) - rho) / h;
            let drho = rho_grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(drho, drho_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_liquid_density_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad =
            EPcpSaftPure::<f64, 4>::seed_derivatives(&pcsaft.0, ["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let pressure = BAR;
        let rho = LiquidDensity(temperature, pressure).evaluate(&pcsaft_ad)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{rho:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = EPcpSaftPure(params);
            let rho_h = State::new_npt(&pcsaft_h, temperature, pressure, (), Some(Liquid))?.density;
            let drho_h = (rho_h.convert_into(MOL / LITER) - rho) / h;
            let drho = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(drho, drho_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_bubble_point_pressure() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad =
            EPcpSaftBinary::<f64, 8>::seed_derivatives(&flat_binary_params(&pcsaft), ["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let x = 0.5;
        let p = BubblePointPressure(temperature, x, None).evaluate(&pcsaft_ad)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = EPcpSaftBinary::new(params, kij);
        let p_h = PhaseEquilibrium::bubble_point(
            &pcsaft_h,
            temperature,
            x,
            None,
            None,
            Default::default(),
        )?
        .vapor()
        .pressure(Contributions::Total);
        let dp_h = (p_h.convert_into(BAR) - p) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dp_h,
            grad,
            ((dp_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dp_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_dew_point_pressure() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad =
            EPcpSaftBinary::<f64, 8>::seed_derivatives(&flat_binary_params(&pcsaft), ["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let y = 0.5;
        let p = DewPointPressure(temperature, y, None).evaluate(&pcsaft_ad)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = EPcpSaftBinary::new(params, kij);
        let p_h =
            PhaseEquilibrium::dew_point(&pcsaft_h, temperature, y, None, None, Default::default())?
                .vapor()
                .pressure(Contributions::Total);
        let dp_h = (p_h.convert_into(BAR) - p) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dp_h,
            grad,
            ((dp_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dp_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_bubble_point_temperature() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad =
            EPcpSaftBinary::<f64, 8>::seed_derivatives(&flat_binary_params(&pcsaft), ["k_ij"]);
        let pressure = Pressure::from_reduced(DualVec::from(45. * BAR.into_reduced()));
        let t_init = Temperature::from_reduced(DualVec::from(500.0));
        let x = DualVec::from(0.5);
        let t = PhaseEquilibrium::bubble_point(
            &pcsaft_ad,
            pressure,
            x,
            Some(t_init),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let t = t.convert_into(KELVIN);
        let (t, [[grad]]) = (t.re, t.eps.unwrap_generic(U1, U1).data.0);

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = EPcpSaftBinary::new(params, kij);
        let t_h = PhaseEquilibrium::bubble_point(
            &pcsaft_h,
            pressure.re(),
            x.re(),
            Some(t_init.re()),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let dt_h = (t_h.convert_into(KELVIN) - t) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dt_h,
            grad,
            ((dt_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dt_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_dew_point_temperature() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad =
            EPcpSaftBinary::<f64, 8>::seed_derivatives(&flat_binary_params(&pcsaft), ["k_ij"]);
        let pressure = Pressure::from_reduced(DualVec::from(45. * BAR.into_reduced()));
        let t_init = Temperature::from_reduced(DualVec::from(500.0));
        let x = DualVec::from(0.5);
        let t = PhaseEquilibrium::dew_point(
            &pcsaft_ad,
            pressure,
            x,
            Some(t_init),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let t = t.convert_into(KELVIN);
        let (t, [[grad]]) = (t.re, t.eps.unwrap_generic(U1, U1).data.0);

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = EPcpSaftBinary::new(params, kij);
        let t_h = PhaseEquilibrium::dew_point(
            &pcsaft_h,
            pressure.re(),
            x.re(),
            Some(t_init.re()),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let dt_h = (t_h.convert_into(KELVIN) - t) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dt_h,
            grad,
            ((dt_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dt_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_bubble_point_temperature_derivative() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad = pcsaft;
        let mut temperature = 500.0 * KELVIN;
        let x = 0.5;
        let (p, grad) = first_derivative(
            partial(
                |t, &x: &Dual64| {
                    let eos = pcsaft_ad.lift();
                    PhaseEquilibrium::bubble_point(&eos, t, x, None, None, Default::default())
                        .unwrap()
                        .vapor()
                        .pressure(Contributions::Total)
                },
                &x,
            ),
            temperature,
        );

        println!("{p:.5}");
        println!("{grad:.5?}");

        let h = 1e-6 * KELVIN;
        temperature += h;
        let p_h = PhaseEquilibrium::bubble_point(
            &pcsaft_ad,
            temperature,
            x,
            None,
            None,
            Default::default(),
        )?
        .vapor()
        .pressure(Contributions::Total);
        let dp_h = (p_h - p) / h;
        println!(
            "{:11.5?} {:11.5?} {:.3e}",
            dp_h,
            grad,
            ((dp_h - grad).convert_into(grad)).abs()
        );
        assert_relative_eq!(grad, dp_h, max_relative = 2e-7);
        Ok(())
    }

    #[test]
    fn test_bubble_point_pressure_derivative() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let pcsaft_ad = pcsaft;
        let mut pressure = 45. * BAR;
        let t0 = Some(500. * KELVIN);
        let x = 0.5;
        let (t, grad) = first_derivative(
            partial2(
                |p, &x: &Dual64, &t0| {
                    let eos = pcsaft_ad.lift();
                    PhaseEquilibrium::bubble_point(&eos, p, x, t0, None, Default::default())
                        .unwrap()
                        .vapor()
                        .temperature
                },
                &x,
                &t0,
            ),
            pressure,
        );

        println!("{t:.5}");
        println!("{grad:.5?}");

        let h = 1e-5 * BAR;
        pressure += h;
        let t_h =
            PhaseEquilibrium::bubble_point(&pcsaft_ad, pressure, x, t0, None, Default::default())?
                .vapor()
                .temperature;
        let dt_h = (t_h - t) / h;
        println!(
            "{:11.5?} {:11.5?} {:.3e}",
            dt_h,
            grad,
            ((dt_h - grad).convert_into(grad)).abs()
        );
        assert_relative_eq!(grad, dt_h, max_relative = 2e-7);
        Ok(())
    }

    #[test]
    fn test_tp_flash() -> FeosResult<()> {
        let (pcsaft, _) = epcpsaft_binary()?;
        let (params, mut kij) = pcsaft.0;
        let mut flat_params: Vec<f64> = params[0].to_vec();
        flat_params.extend_from_slice(&params[1]);
        flat_params.push(kij);
        let pcsaft_ad = EPcpSaftBinary::<f64, 8>::seed_derivatives(&flat_params, ["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let pressure = 44.6 * BAR;
        let x = 0.5;
        let vle = PhaseEquilibrium::tp_flash_binary(
            &pcsaft_ad,
            Temperature::from_inner(&temperature),
            Pressure::from_inner(&pressure),
            DualVec::from(x),
            SolverOptions {
                verbosity: feos_core::Verbosity::Iter,
                tol: Some(1e-12),
                ..Default::default()
            },
        )?;
        let beta = vle.vapor_phase_fraction();
        let (beta, [[grad]]) = (beta.re, beta.eps.unwrap_generic(U1, U1).data.0);

        println!("{beta:.5}");
        println!("{grad:.5?}");
        let h = 1e-7;
        kij += h;
        let pcsaft_h = EPcpSaftBinary::new(params, kij);
        let vle = PhaseEquilibrium::tp_flash_binary(
            &pcsaft_h,
            temperature,
            pressure,
            x,
            SolverOptions {
                tol: Some(1e-12),
                ..Default::default()
            },
        )?;
        let beta_h = vle.vapor_phase_fraction();
        let dbeta_h = (beta_h - beta) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dbeta_h,
            grad,
            ((dbeta_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dbeta_h, max_relative = 1e-4);
        Ok(())
    }
}

#[cfg(test)]
mod tests_electrolyte {
    use super::*;
    use crate::epcpsaft::parameters::utils::{water_nacl_parameters, water_nacl_records};
    use crate::epcpsaft::{EPcpSaftAssociationRecord, EPcpSaftRecord};
    use approx::assert_relative_eq;
    #[cfg(feature = "pcsaft")]
    use feos_core::parameter::IdentifierOption;
    use feos_core::{Contributions, DensityInitialization, FeosError, State};
    use nalgebra::dvector;
    use num_dual::Dual64;
    use quantity::{BAR, COULOMB, JOULE, KB, KELVIN, MOL, QE};

    #[cfg(feature = "pcsaft")]
    const COMPONENTS: [&str; 4] = ["water", "acetone", "methanol", "carbon dioxide"];

    /// ePCP-SAFT without ions is PCP-SAFT: all contributions, pressure and fugacity
    /// coefficients agree for the PCP-SAFT parameters of Esper et al. (2023) and
    /// Rehner et al. (2023).
    #[cfg(feature = "pcsaft")]
    #[test]
    fn neutral_identical_to_pcsaft() -> FeosResult<()> {
        use crate::pcsaft::{PcSaft, PcSaftParameters};
        let files = (
            "../../parameters/pcsaft/esper2023.json",
            Some("../../parameters/pcsaft/rehner2023_binary.json"),
        );
        let pcsaft = PcSaft::new(PcSaftParameters::from_json(
            COMPONENTS.to_vec(),
            files.0,
            files.1,
            IdentifierOption::Name,
        )?);
        let epcpsaft = EPcpSaft::new(EPcpSaftParameters::from_json(
            COMPONENTS.to_vec(),
            files.0,
            files.1,
            IdentifierOption::Name,
        )?)?;
        let x = dvector![0.4, 0.3, 0.2, 0.1];
        for (t, v) in [(300.0, 90.0), (350.0, 120.0), (450.0, 1.0e4)] {
            assert!(
                epcpsaft
                    .reduced_helmholtz_energy_density_contributions(&StateHD::new(t, v, &x))
                    .iter()
                    .all(|(_, a)| a.is_finite())
            );
            let s = StateHD::new(t, v, &x);
            let a = pcsaft.reduced_helmholtz_energy_density_contributions(&s);
            let b = epcpsaft.reduced_helmholtz_energy_density_contributions(&s);
            assert_eq!(a.len(), b.len());
            for ((na, va), (nb, vb)) in a.iter().zip(b.iter()) {
                assert_eq!(na, nb);
                assert_relative_eq!(va, vb, max_relative = 1e-14);
            }
        }
        let n = dvector![0.4, 0.3, 0.2, 0.1] * MOL;
        let sa = State::new_npt(
            &&pcsaft,
            320.0 * KELVIN,
            5.0 * BAR,
            &n,
            Some(DensityInitialization::Liquid),
        )?;
        let sb = State::new_npt(
            &&epcpsaft,
            320.0 * KELVIN,
            5.0 * BAR,
            &n,
            Some(DensityInitialization::Liquid),
        )?;
        assert_relative_eq!(sa.density, sb.density, max_relative = 1e-12);
        for i in 0..4 {
            assert_relative_eq!(sa.ln_phi()[i], sb.ln_phi()[i], max_relative = 1e-12);
        }
        Ok(())
    }

    /// Dipole, quadrupole and dipole-quadrupole terms (both DQ variants) identical to PCP-SAFT
    /// (the PCP-SAFT parameter files contain no quadrupole moments).
    #[cfg(feature = "pcsaft")]
    #[test]
    fn polar_identical_to_pcsaft() -> FeosResult<()> {
        use crate::pcsaft::{self, PcSaft, PcSaftOptions, PcSaftParameters};
        let json = r#"[
            {"identifier": {"name": "co2"}, "molarweight": 44.0, "m": 1.5131, "sigma": 3.1869, "epsilon_k": 163.33, "q": 4.4},
            {"identifier": {"name": "acetone"}, "molarweight": 58.08, "m": 2.7447, "sigma": 3.2742, "epsilon_k": 232.99, "mu": 2.88},
            {"identifier": {"name": "polar"}, "molarweight": 60.0, "m": 2.0, "sigma": 3.5, "epsilon_k": 250.0, "mu": 1.5, "q": 2.0}
        ]"#;
        let x = dvector![0.5, 0.3, 0.2];
        for (dq_pcsaft, dq_epcpsaft) in [
            (pcsaft::DQVariants::DQ35, DQVariants::DQ35),
            (pcsaft::DQVariants::DQ44, DQVariants::DQ44),
        ] {
            let pcsaft = PcSaft::with_options(
                PcSaftParameters::new(serde_json::from_str(json).unwrap(), vec![])?,
                PcSaftOptions {
                    dq_variant: dq_pcsaft,
                    ..Default::default()
                },
            );
            let epcpsaft = EPcpSaft::with_options(
                EPcpSaftParameters::new(serde_json::from_str(json).unwrap(), vec![])?,
                EPcpSaftOptions {
                    dq_variant: dq_epcpsaft,
                    ..Default::default()
                },
            )?;
            for (t, v) in [(250.0, 60.0), (350.0, 1.0e3)] {
                let s = StateHD::new(t, v, &x);
                let a = pcsaft.reduced_helmholtz_energy_density_contributions(&s);
                let b = epcpsaft.reduced_helmholtz_energy_density_contributions(&s);
                assert_eq!(a.len(), 6);
                assert_eq!(a, b);
            }
        }
        Ok(())
    }

    /// Independent check of the ion terms: ePC-SAFT (advanced, crate::epcsaft) with the same
    /// records. The ionic and Born contributions differ only by the vacuum permittivity
    /// (epcsaft: 8.85416e-12 F/m, here 8.8541878188e-12 F/m: relative difference 3.2e-6).
    #[cfg(feature = "epcsaft")]
    #[test]
    fn ion_terms_as_epcsaft() -> FeosResult<()> {
        use crate::epcsaft::{ElectrolytePcSaft, ElectrolytePcSaftParameters};
        let json = serde_json::to_string(&water_nacl_records()).unwrap();
        let epcsaft = ElectrolytePcSaft::new(ElectrolytePcSaftParameters::new(
            serde_json::from_str(&json).unwrap(),
            vec![],
        )?)?;
        let epcpsaft = EPcpSaft::new(water_nacl_parameters())?;
        for (t, x) in [(298.15, [0.9, 0.05, 0.05]), (350.0, [0.98, 0.01, 0.01])] {
            let s = StateHD::new(t, 30.0, &DVector::from_row_slice(&x));
            let a = epcsaft.reduced_helmholtz_energy_density_contributions(&s);
            let b = epcpsaft.reduced_helmholtz_energy_density_contributions(&s);
            for name in ["Ionic", "Born"] {
                let find = |c: &[(&str, f64)]| c.iter().find(|(n, _)| *n == name).unwrap().1;
                assert_relative_eq!(find(&a), find(&b), max_relative = 1e-5);
            }
        }
        Ok(())
    }

    /// parameters/epcpsaft: PCP-SAFT entries identical to parameters/pcsaft, only permittivity records added;
    /// all records can be read as ePCP-SAFT records.
    #[test]
    fn parameter_files() -> FeosResult<()> {
        use feos_core::parameter::PureRecord;
        let read = |f: &str| -> Vec<serde_json::Value> {
            serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap()
        };
        let pcp = read("../../parameters/pcsaft/esper2023.json");
        let mut epcp = read("../../parameters/epcpsaft/esper2023.json");
        assert_eq!(pcp.len(), epcp.len());
        let mut n_permittivity = 0;
        for (a, b) in pcp.iter().zip(epcp.iter_mut()) {
            n_permittivity += b
                .as_object_mut()
                .unwrap()
                .remove("permittivity_record")
                .is_some() as usize;
            assert_eq!(a, b);
        }
        assert!(n_permittivity > 0);
        let records: Vec<PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord>> =
            serde_json::from_str(
                &std::fs::read_to_string("../../parameters/epcpsaft/esper2023.json").unwrap(),
            )?;
        assert_eq!(
            records
                .iter()
                .filter(|r| r.model_record.permittivity_record.is_some())
                .count(),
            n_permittivity
        );
        assert_eq!(
            read("../../parameters/pcsaft/rehner2023_binary.json"),
            read("../../parameters/epcpsaft/rehner2023_binary.json")
        );
        Ok(())
    }

    /// The permittivity record does not change the Helmholtz energy of a system without ions.
    #[test]
    fn permittivity_record_without_ions() -> FeosResult<()> {
        let mut record = water_nacl_records()[0].clone();
        let water = EPcpSaft::new(EPcpSaftParameters::new_pure(record.clone())?)?;
        record.model_record.permittivity_record = None;
        let water_no_eps = EPcpSaft::new(EPcpSaftParameters::new_pure(record)?)?;
        let s = StateHD::new(298.15, 18.0, &dvector![1.0]);
        assert_eq!(
            water.reduced_helmholtz_energy_density_contributions(&s),
            water_no_eps.reduced_helmholtz_energy_density_contributions(&s)
        );
        assert!(water_no_eps.permittivity(&s).is_none());
        assert!(water.permittivity(&s).is_some());
        Ok(())
    }

    #[test]
    fn ion_rules() -> FeosResult<()> {
        let eos = EPcpSaft::new(water_nacl_parameters())?;
        let p = &eos.params;
        assert_eq!(p.ionic_comp, vec![1, 2]);
        // like-charged ions: k_ij = 1, i.e. no dispersion
        assert_eq!(p.epsilon_k_ij[(1, 1)], 0.0);
        assert_eq!(p.epsilon_k_ij[(2, 2)], 0.0);
        assert_eq!(p.epsilon_k_ij[(1, 2)], (230.0f64 * 170.0).sqrt());
        // ion diameter 0.88 sigma, independent of temperature
        for t in [273.15, 373.15] {
            let d = p.hs_diameter(t);
            assert_eq!(d[1], 2.8232 * 0.88);
            assert_eq!(d[2], 2.7560 * 0.88);
            assert_relative_eq!(
                d[0],
                2.15072 * (1.0 - 0.12 * (-3.0 * 230.71557 / t).exp()),
                max_relative = 1e-15
            );
        }
        // Born diameter
        let options = EPcpSaftOptions {
            born_delta_cation: 0.3,
            born_delta_anion: -0.1,
            ..Default::default()
        };
        let eos_delta = EPcpSaft::with_options(water_nacl_parameters(), options)?;
        let db = eos_delta.born_diameter(298.15);
        let d = p.hs_diameter(298.15);
        assert_eq!(db[0], d[0]);
        assert_eq!(db[1], d[1] + 0.3);
        assert_eq!(db[2], d[2] - 0.1);
        Ok(())
    }

    #[test]
    fn missing_permittivity_record() {
        let mut records = water_nacl_records();
        records[0].model_record.permittivity_record = None;
        let p = EPcpSaftParameters::new(records, vec![]).unwrap();
        assert!(matches!(
            EPcpSaft::new(p),
            Err(FeosError::IncompatibleParameters(_))
        ));
    }

    /// d(A_Born / kT) / dN_i at infinite dilution of the ions is the Born solvation energy
    /// -z_i^2 e^2 / (4 pi eps_0 kT d_i) (1 - 1/eps_r(pure water)).
    #[test]
    fn born_infinite_dilution() -> FeosResult<()> {
        let eos = EPcpSaft::new(water_nacl_parameters())?;
        let t = 298.15;
        let v = 30.0;
        let n = dvector![
            Dual64::from(1.0),
            Dual64::from(0.0).derivative(),
            Dual64::from(0.0)
        ];
        let s = StateHD::new(Dual64::from(t), Dual64::from(v), &n);
        let contributions = eos.reduced_helmholtz_energy_density_contributions(&s);
        let (_, born) = contributions
            .iter()
            .find(|(name, _)| *name == "Born")
            .unwrap();
        let mu_born = (*born * v).eps;

        let eps = eos
            .permittivity(&StateHD::new(t, v, &dvector![1.0, 0.0, 0.0]))
            .unwrap();
        let e2_kt = (QE / COULOMB).into_value().powi(2)
            / (4.0
                * std::f64::consts::PI
                * permittivity::epsilon_0()
                * (KB / (JOULE / KELVIN)).into_value()
                * t)
            * 1e10;
        let d_na = 2.8232 * 0.88;
        assert_relative_eq!(
            mu_born,
            -e2_kt / d_na * (1.0 - 1.0 / eps),
            max_relative = 1e-12
        );
        Ok(())
    }

    /// d(A_DH / kT) / dN_i approaches the Debye-Hückel limiting law -z_i^2 lambda_B kappa / 2.
    #[test]
    fn debye_hueckel_limiting_law() -> FeosResult<()> {
        for (variant, packing, name) in VARIANTS {
            limiting_law(variant, packing, name)?;
        }
        Ok(())
    }

    /// all ionic terms
    const VARIANTS: [(EPcpSaftVariants, MsaPacking, &str); 3] = [
        (EPcpSaftVariants::DebyeHueckel, MsaPacking::Ions, "Ionic"),
        (
            EPcpSaftVariants::MeanSphericalApproximation,
            MsaPacking::Ions,
            "MSA",
        ),
        (
            EPcpSaftVariants::MeanSphericalApproximation,
            MsaPacking::Total,
            "MSA",
        ),
    ];

    fn with_variant(
        epcpsaft_variant: EPcpSaftVariants,
        msa_packing: MsaPacking,
    ) -> FeosResult<EPcpSaft> {
        let options = EPcpSaftOptions {
            epcpsaft_variant,
            msa_packing,
            ..Default::default()
        };
        EPcpSaft::with_options(water_nacl_parameters(), options)
    }

    fn limiting_law(variant: EPcpSaftVariants, packing: MsaPacking, name: &str) -> FeosResult<()> {
        let eos = with_variant(variant, packing)?;
        let t = 298.15;
        let v = 30.0;
        // deviations from the limiting law are of order kappa d (6e-4, 6e-7, 6e-10)
        for (c, tol) in [(1e-8, 1e-3), (1e-14, 1e-6), (1e-20, 1e-9)] {
            let n = dvector![
                Dual64::from(1.0),
                Dual64::from(c).derivative(),
                Dual64::from(c)
            ];
            let s = StateHD::new(Dual64::from(t), Dual64::from(v), &n);
            let contributions = eos.reduced_helmholtz_energy_density_contributions(&s);
            let (_, ionic) = contributions.iter().find(|(n, _)| *n == name).unwrap();
            let mu_dh = (*ionic * v).eps;

            let s = StateHD::new(t, v, &dvector![1.0, c, c]);
            let eps = eos.permittivity(&s).unwrap();
            let lambda_b = eos.params.bjerrum_length(t, eps);
            let kappa = (4.0 * std::f64::consts::PI * lambda_b * 2.0 * c / v).sqrt();
            assert_relative_eq!(mu_dh, -lambda_b * kappa / 2.0, max_relative = tol);
        }
        Ok(())
    }

    /// Residual pressure of all contributions (via automatic differentiation) agrees with a
    /// central finite difference of the Helmholtz energy, including the density dependence
    /// of the permittivity in the ionic and Born terms.
    #[test]
    fn pressure_finite_difference() -> FeosResult<()> {
        for (variant, packing, _) in VARIANTS {
            pressure_fd(variant, packing)?;
        }
        Ok(())
    }

    fn pressure_fd(variant: EPcpSaftVariants, packing: MsaPacking) -> FeosResult<()> {
        let eos = with_variant(variant, packing)?;
        let t = 310.0;
        let v = 25.0;
        let n = dvector![0.9, 0.05, 0.05];
        let a = |v: f64| {
            eos.reduced_helmholtz_energy_density_contributions(&StateHD::new(t, v, &n))
                .into_iter()
                .map(|(name, a)| (name, a * v))
                .collect::<Vec<_>>()
        };
        let h = 1e-5;
        let (ap, am) = (a(v + h), a(v - h));
        let vd = Dual64::from(v).derivative();
        let s = StateHD::new(Dual64::from(t), vd, &n.map(Dual64::from));
        let ad = eos.reduced_helmholtz_energy_density_contributions(&s);
        for (((name, a_ad), (_, a_p)), (_, a_m)) in ad.iter().zip(ap.iter()).zip(am.iter()) {
            let fd = (a_p - a_m) / (2.0 * h);
            assert_relative_eq!((*a_ad * vd).eps, fd, max_relative = 1e-7);
            println!("{name}: {} {fd}", (*a_ad * vd).eps);
        }
        Ok(())
    }

    /// MSA and Debye-Hückel agree in the limit of low concentrations (both -κ³/(12π)) and differ
    /// at finite concentrations; all other contributions are identical
    #[test]
    fn msa_and_debye_hueckel() -> FeosResult<()> {
        let dh = with_variant(EPcpSaftVariants::DebyeHueckel, MsaPacking::Ions)?;
        let msa = with_variant(
            EPcpSaftVariants::MeanSphericalApproximation,
            MsaPacking::Ions,
        )?;
        for (c, close) in [(1e-9, true), (0.05, false)] {
            let s = StateHD::new(298.15, 30.0, &dvector![1.0, c, c]);
            let a = dh.reduced_helmholtz_energy_density_contributions(&s);
            let b = msa.reduced_helmholtz_energy_density_contributions(&s);
            for ((na, va), (nb, vb)) in a.iter().zip(&b) {
                if *na == "Ionic" {
                    assert_eq!(*nb, "MSA");
                    let rel = (vb / va - 1.0).abs();
                    assert!(if close { rel < 1e-3 } else { rel > 1e-2 }, "{c}: {rel}");
                } else {
                    assert_eq!(na, nb);
                    assert_eq!(va, vb);
                }
            }
        }
        Ok(())
    }

    /// Gibbs-Duhem relation at constant T and p, sum_i x_i d ln(phi_i) = 0, for all variants:
    /// (1) with the composition derivatives N (d ln phi_i / d N_j) of feos, (2) with central
    /// differences of ln phi between electroneutral compositions (from 0.1 mol/kg).
    #[test]
    fn gibbs_duhem() -> FeosResult<()> {
        for (variant, packing, _) in VARIANTS {
            let eos = with_variant(variant, packing)?;
            for t in [298.15, 373.15] {
                for m in [0.01, 0.1, 1.0, 5.0] {
                    let n = |m: f64| dvector![55.508, m, m] * MOL;
                    let state = |m: f64| {
                        State::new_npt(
                            &&eos,
                            t * KELVIN,
                            5.0 * BAR,
                            n(m),
                            Some(DensityInitialization::Liquid),
                        )
                    };
                    let s = state(m)?;
                    let x = s.molefracs.clone();
                    let gd = s.n_dln_phi_dnj().transpose() * &x;
                    assert!(gd.amax() < 1e-10, "{variant:?} {packing:?} {t} {m}: {gd}");

                    // step 1e-3 m: truncation error of order 1e-6 relative; the convergence tolerance
                    // of the density iteration limits ln phi to about 1e-11, which exceeds the terms
                    // x_i d ln(phi_i) (about 1e-8) below 0.1 mol/kg
                    if m < 0.1 {
                        continue;
                    }
                    let h = 1e-3 * m;
                    let dln_phi = state(m + h)?.ln_phi() - state(m - h)?.ln_phi();
                    let terms = x.component_mul(&dln_phi);
                    let (sum, scale) = (terms.sum(), terms.abs().sum());
                    assert!(
                        (sum / scale).abs() < 1e-4,
                        "{variant:?} {packing:?} {t} {m}: {sum} {scale}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn electrolyte_state() -> FeosResult<()> {
        let eos = EPcpSaft::new(water_nacl_parameters())?;
        let n = dvector![0.9, 0.05, 0.05] * MOL;
        let s = State::new_npt(
            &&eos,
            298.15 * KELVIN,
            BAR,
            &n,
            Some(DensityInitialization::Liquid),
        )?;
        let p = s.pressure(Contributions::Total);
        assert_relative_eq!(p, BAR, max_relative = 1e-8);
        Ok(())
    }
}
