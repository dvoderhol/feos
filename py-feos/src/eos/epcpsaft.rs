use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyGcParameters, PyParameters};
use crate::residual::ResidualModel;
use crate::state::PyState;
#[cfg(feature = "dft")]
use feos::epcpsaft::EPcpSaftFunctional;
use feos::epcpsaft::{DQVariants, EPcpSaft, EPcpSaftOptions};
use feos_core::{EquationOfState, ReferenceSystem, ResidualDyn, StateHD};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

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
    /// ePCP-SAFT equation of state: PCP-SAFT with Debye-Hückel and Born contributions
    /// and the permittivity model of Rueben et al. (2024).
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
    /// born_delta_cation : float, optional
    ///     Correction added to the diameter of all cations in the Born term (in Angstrom). Defaults to 0.
    /// born_delta_anion : float, optional
    ///     Correction added to the diameter of all anions in the Born term (in Angstrom). Defaults to 0.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The ePCP-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35", born_delta_cation=0.0, born_delta_anion=0.0),
        text_signature = r#"(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35", born_delta_cation=0.0, born_delta_anion=0.0)"#
    )]
    pub fn epcpsaft(
        parameters: &Bound<'_, PyAny>,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
        born_delta_cation: f64,
        born_delta_anion: f64,
    ) -> PyResult<Self> {
        let options = EPcpSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant: self::dq_variant(dq_variant)?,
            born_delta_cation,
            born_delta_anion,
        };
        let parameters = if let Ok(parameters) = parameters.extract::<PyParameters>() {
            parameters.try_convert()
        } else if let Ok(parameters) = parameters.extract::<PyGcParameters>() {
            parameters.try_convert_homosegmented()
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "Argument `parameters` must by Parameters or GcParameters",
            ));
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
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35"),
        text_signature = r#"(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")"#
    )]
    fn epcpsaft(
        parameters: PyParameters,
        fmt_version: PyFMTVersion,
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
    #[pyo3(text_signature = "($self)")]
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
        let state = StateHD::new_density(s.temperature.into_reduced(), &partial_density);
        eos.permittivity(&state)
            .ok_or_else(|| PyErr::new::<PyValueError, _>("A component has no permittivity record."))
    }
}
