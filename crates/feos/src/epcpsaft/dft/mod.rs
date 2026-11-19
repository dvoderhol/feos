use super::EPcpSaftParameters;
use super::parameters::EPcpSaftPars;
use crate::association::{AssociationVersion, SaftAssociationFunctional};
use crate::epcpsaft::eos::EPcpSaftOptions;
use crate::hard_sphere::{FMTContribution, FMTVersion};
use feos_core::{
    FeosError, FeosResult, Molarweight, ReferenceSystem, ResidualDyn, StateHD, Subset,
};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, HelmholtzEnergyFunctionalDyn, MoleculeShape,
};
use nalgebra::DVector;
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use num_traits::One;
use quantity::{Energy, MolarWeight};
use std::f64::consts::FRAC_PI_6;

mod dispersion;
mod hard_chain;
mod polar;
use dispersion::AttractiveFunctional;
use hard_chain::ChainFunctional;

/// ePCP-SAFT Helmholtz energy functional (currently only for systems without ions).
pub struct EPcpSaftFunctional {
    pub parameters: EPcpSaftParameters,
    params: EPcpSaftPars,
    fmt_version: FMTVersion,
    association_version: AssociationVersion,
    options: EPcpSaftOptions,
}

impl EPcpSaftFunctional {
    pub fn new(parameters: EPcpSaftParameters) -> FeosResult<Self> {
        Self::with_options(
            parameters,
            FMTVersion::WhiteBear,
            AssociationVersion::ISaft,
            EPcpSaftOptions::default(),
        )
    }

    pub fn with_options(
        parameters: EPcpSaftParameters,
        fmt_version: FMTVersion,
        association_version: AssociationVersion,
        saft_options: EPcpSaftOptions,
    ) -> FeosResult<Self> {
        let params = EPcpSaftPars::new(&parameters)?;
        if params.nionic > 0 {
            return Err(FeosError::IncompatibleParameters(
                "The ePCP-SAFT Helmholtz energy functional is not implemented for ions."
                    .to_string(),
            ));
        }
        Ok(Self {
            parameters,
            params,
            fmt_version,
            options: saft_options,
            association_version,
        })
    }
}

impl Subset for EPcpSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.fmt_version,
            self.association_version,
            self.options,
        )
        .unwrap()
    }
}

impl ResidualDyn for EPcpSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<Primitive = f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let msigma3 = p.m.zip_map(&p.sigma, |m, s| m * s.powi(3));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn for_each_contribution<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
        f: impl FnMut(&'static str, D),
    ) {
        self.evaluate_bulk(state, f)
    }
}

impl HelmholtzEnergyFunctionalDyn for EPcpSaftFunctional {
    type Contribution<'a>
        = EPcpSaftFunctionalContribution<'a>
    where
        Self: 'a;
    type Association<'a> = SaftAssociationFunctional<'a, EPcpSaftPars>;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = EPcpSaftFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(3);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.params, self.fmt_version);
        contributions.push(hs.into());

        // Hard chains
        if self.params.m.iter().any(|&mi| !mi.is_one()) {
            let chain = ChainFunctional::new(&self.params);
            contributions.push(chain.into());
        }

        // Dispersion
        let att = AttractiveFunctional::new(&self.params);
        contributions.push(att.into());

        contributions.into_iter()
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::NonSpherical(&self.params.m)
    }

    fn association(&self) -> Option<Self::Association<'_>> {
        (!self.parameters.association.is_empty()).then(|| {
            SaftAssociationFunctional::new(
                &self.params,
                &self.parameters.association,
                self.association_version,
            )
        })
    }
}

impl Molarweight for EPcpSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl FluidParameters for EPcpSaftFunctional {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.params.sigma.clone()
    }
}

impl PairPotential for EPcpSaftFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Energy<Array2<f64>> {
        let sigma_ij = &self.params.sigma_ij;
        let eps_ij_4 = 4.0 * &self.params.epsilon_k_ij;
        Energy::from_reduced(Array2::from_shape_fn(
            (self.params.m.len(), r.len()),
            |(j, k)| {
                let att = (sigma_ij[(i, j)] / r[k]).powi(6);
                eps_ij_4[(i, j)] * att * (att - 1.0)
            },
        ))
    }
}

/// Individual contributions for the ePCP-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
#[expect(private_interfaces)]
pub enum EPcpSaftFunctionalContribution<'a> {
    Fmt(FMTContribution<'a, EPcpSaftPars>),
    Chain(ChainFunctional<'a>),
    Attractive(AttractiveFunctional<'a>),
}
