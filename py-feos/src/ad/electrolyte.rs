use super::PyPropertyAD;
use crate::eos::epcpsaft::epcpsaft_options;
use crate::error::PyFeosError;
use crate::parameter::PyParameters;
use feos::epcpsaft::{
    EPcpSaft, EPcpSaftAD, EPcpSaftParameter, ElectrolyteDerivatives,
    electrolyte_properties_derivatives,
};
use nalgebra::DVector;
use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quantity::{KELVIN, MOL, PASCAL};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::sync::Arc;

fn parse_parameter(name: &str, index: &[usize]) -> PyResult<EPcpSaftParameter> {
    use EPcpSaftParameter::*;
    let one = |f: fn(usize) -> EPcpSaftParameter| match index {
        [i] => Ok(f(*i)),
        _ => Err(PyValueError::new_err(format!(
            "parameter `{name}` needs one component index"
        ))),
    };
    match name {
        "m" => one(M),
        "sigma" => one(Sigma),
        "epsilon_k" => one(EpsilonK),
        "mu" => one(Mu),
        "q" => one(Q),
        "dipole_scaling" => one(DipoleScaling),
        "polarizability_scaling" => one(PolarizabilityScaling),
        "correlation_integral_parameter" => one(CorrelationIntegralParameter),
        "k_ij" => match index {
            [i, j] => Ok(KIJ(*i, *j)),
            _ => Err(PyValueError::new_err(
                "parameter `k_ij` needs two component indices",
            )),
        },
        "born_delta_cation" if index.is_empty() => Ok(BornDeltaCation),
        "born_delta_anion" if index.is_empty() => Ok(BornDeltaAnion),
        _ => Err(PyValueError::new_err(format!(
            "unknown parameter `{name}` with indices {index:?}"
        ))),
    }
}

type ElectrolyteResult<'py> = (
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyArray1<bool>>,
);

/// names of the scalar results of [ElectrolyteDerivatives]
const SCALARS: [&str; 4] = [
    "ln_solvent_activity",
    "volume_difference",
    "residual_heat_capacity_difference",
    "residual_enthalpy_difference",
];

#[pymethods]
impl PyPropertyAD {
    /// Properties of electrolyte solutions with ePCP-SAFT and their derivatives with respect to
    /// model parameters, evaluated in parallel for all rows of `input`.
    ///
    /// For every row, the solution and the pure solvent (amount of the solvent of the row, all
    /// other amounts exactly zero) are evaluated at the temperature and pressure of the row.
    ///
    /// Parameters
    /// ----------
    /// parameters : Parameters
    ///     The ePCP-SAFT parameters.
    /// derivatives : list[tuple[str, list[int]]]
    ///     Parameters for which derivatives are calculated, e.g. ``("sigma", [1])``,
    ///     ``("k_ij", [0, 1])``, ``("born_delta_cation", [])``. Names: m, sigma, epsilon_k, mu, q,
    ///     dipole_scaling, polarizability_scaling, correlation_integral_parameter, k_ij,
    ///     born_delta_cation, born_delta_anion.
    /// input : np.ndarray[float]
    ///     One row per state: temperature (K), pressure (Pa), amounts of all components (mol).
    ///     Every row must be electroneutral.
    /// solvent : int, optional
    ///     Index of the solvent. Defaults to 0.
    /// max_eta, max_iter_cross_assoc, tol_cross_assoc, dq_variant, epcpsaft_variant, msa_packing,
    /// born_delta_cation, born_delta_anion : optional
    ///     Options of the equation of state, see `EquationOfState.epcpsaft`.
    ///
    /// Returns
    /// -------
    /// (dict[str, np.ndarray[float]], dict[str, np.ndarray[float]], np.ndarray[bool])
    ///     Values, derivatives and convergence status. Keys: ``ln_solvent_activity``,
    ///     ``volume_difference`` (volume of the solution minus volume of the pure solvent, m³),
    ///     ``residual_heat_capacity_difference`` (J/K), ``residual_enthalpy_difference`` (J), all
    ///     with shape ``[n]``, and ``ln_gamma`` with shape ``[n, n_components]`` (mole-fraction
    ///     scale; solutes: infinite dilution in the pure solvent; solvent: pure solvent). The
    ///     derivatives have an additional last axis of length ``len(derivatives)``. Rows that did
    ///     not converge are NaN.
    #[staticmethod]
    #[pyo3(signature = (parameters, derivatives, input, solvent=0, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35", epcpsaft_variant="dh", msa_packing="ions", born_delta_cation=0.0, born_delta_anion=0.0))]
    #[expect(clippy::too_many_arguments)]
    pub fn electrolyte_properties_derivatives<'py>(
        py: Python<'py>,
        parameters: PyParameters,
        derivatives: Vec<(String, Vec<usize>)>,
        input: PyReadonlyArray2<f64>,
        solvent: usize,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
        epcpsaft_variant: &str,
        msa_packing: &str,
        born_delta_cation: f64,
        born_delta_anion: f64,
    ) -> PyResult<ElectrolyteResult<'py>> {
        let options = epcpsaft_options(
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
            epcpsaft_variant,
            msa_packing,
            born_delta_cation,
            born_delta_anion,
        )?;
        let parameters = parameters.try_convert()?;
        let model =
            Arc::new(EPcpSaft::with_options(parameters, options).map_err(PyFeosError::from)?);
        let derivatives = derivatives
            .iter()
            .map(|(name, index)| parse_parameter(name, index))
            .collect::<PyResult<Vec<_>>>()?;
        EPcpSaftAD::new(model.clone())
            .check_derivatives(&derivatives)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let input = input.as_array().to_owned();
        let z = model.params().z.clone();
        let nc = z.len();
        if input.ncols() != 2 + nc {
            return Err(PyValueError::new_err(format!(
                "input needs 2 + {nc} columns (temperature, pressure, amounts), got {}",
                input.ncols()
            )));
        }
        if solvent >= nc {
            return Err(PyValueError::new_err(format!(
                "solvent index {solvent} out of range ({nc} components)"
            )));
        }
        for (i, row) in input.rows().into_iter().enumerate() {
            let n = row.iter().skip(2);
            let charge: f64 = n.clone().zip(&z).map(|(n, z)| n * z).sum();
            let scale: f64 = n.zip(&z).map(|(n, z)| (n * z).abs()).sum();
            if charge.abs() > 1e-12 * scale {
                return Err(PyValueError::new_err(format!(
                    "row {i} is not electroneutral (sum of n_i z_i = {charge})"
                )));
            }
        }
        let (n, np) = (input.nrows(), derivatives.len());

        let row = |i: usize| {
            let r = input.row(i);
            let moles = DVector::from_iterator(nc, r.iter().skip(2).copied()) * MOL;
            electrolyte_properties_derivatives(
                &model,
                &derivatives,
                r[0] * KELVIN,
                r[1] * PASCAL,
                &moles,
                solvent,
            )
        };
        let results: Vec<_> = py.detach(|| {
            #[cfg(feature = "rayon")]
            let r = (0..n).into_par_iter().map(row).collect();
            #[cfg(not(feature = "rayon"))]
            let r = (0..n).map(row).collect();
            r
        });

        let mut scalars = [(); 4].map(|_| Array1::from_elem(n, f64::NAN));
        let mut scalar_grads = [(); 4].map(|_| Array2::from_elem((n, np), f64::NAN));
        let mut ln_gamma = Array2::from_elem((n, nc), f64::NAN);
        let mut ln_gamma_grad = Array3::from_elem((n, nc, np), f64::NAN);
        let mut status = Array1::from_elem(n, false);
        for (i, result) in results.into_iter().enumerate() {
            let Ok(e): Result<ElectrolyteDerivatives, _> = result else {
                continue;
            };
            status[i] = true;
            let s = [
                &e.ln_solvent_activity,
                &e.volume_difference,
                &e.residual_heat_capacity_difference,
                &e.residual_enthalpy_difference,
            ];
            for (q, vg) in s.iter().enumerate() {
                scalars[q][i] = vg.value;
                for k in 0..np {
                    scalar_grads[q][(i, k)] = vg.gradient[k];
                }
            }
            for (c, vg) in e.ln_gamma.iter().enumerate() {
                ln_gamma[(i, c)] = vg.value;
                for k in 0..np {
                    ln_gamma_grad[(i, c, k)] = vg.gradient[k];
                }
            }
        }
        let values = PyDict::new(py);
        let gradients = PyDict::new(py);
        for (q, name) in SCALARS.iter().enumerate() {
            values.set_item(name, scalars[q].to_pyarray(py))?;
            gradients.set_item(name, scalar_grads[q].to_pyarray(py))?;
        }
        values.set_item("ln_gamma", ln_gamma.to_pyarray(py))?;
        gradients.set_item("ln_gamma", ln_gamma_grad.to_pyarray(py))?;
        Ok((values, gradients, status.to_pyarray(py)))
    }
}
