use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyAssociationVersion, PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyParameters, PyParametersOrGcParameters};
use crate::residual::ResidualModel;
use crate::state::PyState;
#[cfg(feature = "dft")]
use feos::epcpsaft::EPcpSaftFunctional;
use feos::epcpsaft::{
    BiMsaClosure, BornDiameter, DQVariants, EPcpSaft, EPcpSaftOptions, EPcpSaftVariants,
    MsaPacking, PairingConstant,
};
use feos_core::{EquationOfState, ReferenceSystem, ResidualDyn, StateHD};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

fn epcpsaft_variant(epcpsaft_variant: &str) -> PyResult<EPcpSaftVariants> {
    match epcpsaft_variant {
        "dh" => Ok(EPcpSaftVariants::DebyeHueckel),
        "msa" => Ok(EPcpSaftVariants::MeanSphericalApproximation),
        "bidh" => Ok(EPcpSaftVariants::BiDH),
        "bimsa" => Ok(EPcpSaftVariants::BiMSA),
        _ => Err(PyErr::new::<PyValueError, _>(
            r#"epcpsaft_variant must be "dh", "msa", "bidh" or "bimsa""#,
        )),
    }
}

/// ePCP-SAFT options from the keyword arguments shared by all Python constructors.
#[expect(clippy::too_many_arguments)]
pub(crate) fn epcpsaft_options(
    max_eta: f64,
    max_iter_cross_assoc: usize,
    tol_cross_assoc: f64,
    dq_variant: &str,
    epcpsaft_variant: &str,
    msa_packing: &str,
    born_delta_cation: f64,
    born_delta_anion: f64,
    born_diameter: &str,
    dh_delta: f64,
    ion_pairing_constant: &str,
    ion_pairing_shell: f64,
    bidh_dipole_factor: f64,
    bimsa_closure: &str,
) -> PyResult<EPcpSaftOptions> {
    let born_diameter = match born_diameter {
        "hs" => BornDiameter::HardSphere,
        "sigma" => BornDiameter::Sigma,
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                r#"born_diameter must be "hs" or "sigma""#,
            ));
        }
    };
    let ion_pairing_constant = match ion_pairing_constant {
        "bjerrum" => PairingConstant::Bjerrum,
        "ebeling" => PairingConstant::Ebeling,
        "shell" => PairingConstant::Shell(ion_pairing_shell),
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                r#"ion_pairing_constant must be "bjerrum", "ebeling" or "shell""#,
            ));
        }
    };
    let bimsa_closure = match bimsa_closure {
        "variational" => BiMsaClosure::Variational,
        "bernard_blum" => BiMsaClosure::BernardBlum,
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                r#"bimsa_closure must be "variational" or "bernard_blum""#,
            ));
        }
    };
    let msa_packing = match msa_packing {
        "ions" => MsaPacking::Ions,
        "total" => MsaPacking::Total,
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                r#"msa_packing must be "ions" or "total""#,
            ));
        }
    };
    Ok(EPcpSaftOptions {
        max_eta,
        max_iter_cross_assoc,
        tol_cross_assoc,
        dq_variant: self::dq_variant(dq_variant)?,
        epcpsaft_variant: self::epcpsaft_variant(epcpsaft_variant)?,
        msa_packing,
        born_delta_cation,
        born_delta_anion,
        born_diameter,
        dh_delta,
        ion_pairing_constant,
        bidh_dipole_factor,
        bimsa_closure,
    })
}

fn dq_variant(dq_variant: &str) -> PyResult<DQVariants> {
    match dq_variant {
        "dq35" => Ok(DQVariants::DQ35),
        "dq44" => Ok(DQVariants::DQ44),
        _ => Err(PyErr::new::<PyValueError, _>(
            r#"dq_variant must be "dq35" or "dq44""#,
        )),
    }
}

#[pymethods]
impl PyEquationOfState {
    /// ePCP-SAFT equation of state: PCP-SAFT with ion-ion contributions (Debye-Hückel, mean
    /// spherical approximation, or either with ion pairing), a Born contribution and the
    /// permittivity model of Rueben et al. (2024).
    ///
    /// Parameters
    /// ----------
    /// parameters : Parameters
    ///     The parameters of the ePCP-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float, optional
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : "dq35" | "dq44", optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to "dq35"
    /// epcpsaft_variant : "dh" | "msa" | "bidh" | "bimsa", optional
    ///     Term for the electrostatic interactions between ions: Debye-Hückel ("dh"), mean
    ///     spherical approximation of the primitive model ("msa"), Debye-Hückel with ion pairing
    ///     ("bidh") or binding mean spherical approximation ("bimsa"). The ion pairing of a cation
    ///     and an anion uses the association constant of `ion_pairing_constant` times
    ///     exp(`ion_pairing`) of the binary record. Defaults to "dh".
    /// msa_packing : "ions" | "total", optional
    ///     Species in the packing term of the MSA and BiMSA: ions only (primitive model) or all
    ///     species (excluded volume of the solvent included). Defaults to "ions".
    /// born_delta_cation : float, optional
    ///     Correction added to the diameter of all cations in the Born term (in Angstrom). Defaults to 0.
    /// born_delta_anion : float, optional
    ///     Correction added to the diameter of all anions in the Born term (in Angstrom). Defaults to 0.
    /// born_diameter : "hs" | "sigma", optional
    ///     Base of the ion diameters in the Born term: hard-sphere diameter 0.88 σ_i ("hs"; Held et
    ///     al. 2014) or σ_i ("sigma"; Bülow et al. 2021). Defaults to "hs".
    /// dh_delta : float, optional
    ///     Correction added to the diameter of all ions in the Debye-Hückel term of the variant "dh"
    ///     (in Angstrom). Defaults to 0.
    /// ion_pairing_constant : "bjerrum" | "ebeling" | "shell", optional
    ///     Association constant of the ion pairs of the variants "bidh" and "bimsa": Bjerrum
    ///     (zero below l_B |z_i z_j| / d_ij = 2), Ebeling (analytic), or the Bjerrum integral over the
    ///     shell from d_ij to `ion_pairing_shell` times d_ij (contact pairs). Defaults to "bjerrum".
    /// ion_pairing_shell : float, optional
    ///     Binding distance relative to the distance of closest approach for
    ///     `ion_pairing_constant="shell"`. Defaults to 1.034 (Naseri Boroujeni et al. 2024).
    /// bidh_dipole_factor : float, optional
    ///     Factor of the interaction of the dipole of the ion pairs with the ionic atmosphere in the
    ///     variant "bidh" (0: without; 0.25: inner permittivity of the pair 0, Naseri Boroujeni et al.
    ///     2023). Defaults to 0.
    /// bimsa_closure : "variational" | "bernard_blum", optional
    ///     Closure of the variant "bimsa": stationary point of the variational functional, or the
    ///     equations of Bernard and Blum (1996) with the pairs in the asymmetry parameter.
    ///     Defaults to "variational".
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The ePCP-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35", epcpsaft_variant="dh", msa_packing="ions", born_delta_cation=0.0, born_delta_anion=0.0, born_diameter="hs", dh_delta=0.0, ion_pairing_constant="bjerrum", ion_pairing_shell=1.034, bidh_dipole_factor=0.0, bimsa_closure="variational")
    )]
    #[expect(clippy::too_many_arguments)]
    pub fn epcpsaft(
        parameters: PyParametersOrGcParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
        epcpsaft_variant: &str,
        msa_packing: &str,
        born_delta_cation: f64,
        born_delta_anion: f64,
        born_diameter: &str,
        dh_delta: f64,
        ion_pairing_constant: &str,
        ion_pairing_shell: f64,
        bidh_dipole_factor: f64,
        bimsa_closure: &str,
    ) -> PyResult<Self> {
        let options = epcpsaft_options(
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
            epcpsaft_variant,
            msa_packing,
            born_delta_cation,
            born_delta_anion,
            born_diameter,
            dh_delta,
            ion_pairing_constant,
            ion_pairing_shell,
            bidh_dipole_factor,
            bimsa_closure,
        )?;
        let parameters = match parameters {
            PyParametersOrGcParameters::Parameters(parameters) => parameters.try_convert(),
            PyParametersOrGcParameters::GcParameters(parameters) => {
                parameters.try_convert_homosegmented()
            }
        }?;
        let residual = ResidualModel::EPcpSaft(
            EPcpSaft::with_options(parameters, options).map_err(PyFeosError::from)?,
        );
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}

#[cfg(feature = "dft")]
#[pymethods]
impl PyHelmholtzEnergyFunctional {
    /// ePCP-SAFT Helmholtz energy functional (systems without ions only).
    ///
    /// Parameters
    /// ----------
    /// parameters: Parameters
    ///     The set of ePCP-SAFT parameters.
    /// fmt_version: FMTVersion, optional
    ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
    /// association_version: AssociationVersion, optional
    ///     The version of the association functional. Defaults to AssociationVersion.ISaft
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : "dq35" | "dq44", optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to "dq35"
    ///
    /// Returns
    /// -------
    /// HelmholtzEnergyFunctional
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, association_version=PyAssociationVersion::ISaft, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")
    )]
    fn epcpsaft(
        parameters: PyParameters,
        fmt_version: PyFMTVersion,
        association_version: PyAssociationVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
    ) -> PyResult<PyEquationOfState> {
        let options = EPcpSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant: self::dq_variant(dq_variant)?,
            ..Default::default()
        };
        let func = ResidualModel::EPcpSaftFunctional(
            EPcpSaftFunctional::with_options(
                parameters.try_convert()?,
                fmt_version.into(),
                association_version.into(),
                options,
            )
            .map_err(PyFeosError::from)?,
        );
        let ideal_gas = vec![IdealGasModel::NoModel; func.components()];
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}

#[pymethods]
impl PyState {
    /// Relative permittivity of the state (ePCP-SAFT only).
    ///
    /// Returns
    /// -------
    /// float
    fn permittivity(&self) -> PyResult<f64> {
        let s = &self.0;
        let eos = match &s.eos.residual {
            ResidualModel::EPcpSaft(eos) => eos,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    "The permittivity is only available for ePCP-SAFT.",
                ));
            }
        };
        let partial_density = &s.molefracs * s.density.into_reduced();
        let state = StateHD::new_density(s.temperature.into_reduced(), partial_density);
        eos.permittivity(&state)
            .ok_or_else(|| PyErr::new::<PyValueError, _>("A component has no permittivity record."))
    }
}
