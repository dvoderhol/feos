use feos_core::parameter::{CombiningRule, FromSegments, FromSegmentsBinary, Parameters};
use feos_core::{FeosError, FeosResult};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_traits::Zero;
use quantity::{JOULE, KB, KELVIN};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

/// Parameters of the permittivity model of Rueben et al. (2024), perturbation theory of Neumaier et al. (2022).
///
/// Same JSON format as `PermittivityRecord::PerturbationTheory` of ePC-SAFT.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum PermittivityRecord {
    PerturbationTheory {
        /// Scaled squared dipole moment (a11 mu^2) in units of Debye^2
        dipole_scaling: f64,
        /// Scaled polarizability (a12 alpha) in units of Angstrom^3
        polarizability_scaling: f64,
        /// Parameter of the correlation integral (a2)
        correlation_integral_parameter: f64,
    },
}

/// ePCP-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EPcpSaftRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub mu: f64,
    /// Quadrupole moment in units of Debye * Angstrom
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub q: f64,
    /// Charge number
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub z: f64,
    /// Parameters of the permittivity model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permittivity_record: Option<PermittivityRecord>,
    /// Entropy scaling coefficients for the viscosity
    #[serde(skip_serializing_if = "Option::is_none")]
    viscosity: Option<[f64; 4]>,
    /// Entropy scaling coefficients for the diffusion coefficient
    #[serde(skip_serializing_if = "Option::is_none")]
    diffusion: Option<[f64; 5]>,
    /// Entropy scaling coefficients for the thermal conductivity
    #[serde(skip_serializing_if = "Option::is_none")]
    thermal_conductivity: Option<[f64; 4]>,
}

impl FromSegments for EPcpSaftRecord {
    fn from_segments(segments: &[(Self, f64)]) -> FeosResult<Self> {
        let mut m = 0.0;
        let mut sigma3 = 0.0;
        let mut epsilon_k = 0.0;
        let mut mu = 0.0;
        let mut q = 0.0;
        let mut z = 0.0;

        segments.iter().for_each(|(s, n)| {
            m += s.m * n;
            sigma3 += s.m * s.sigma.powi(3) * n;
            epsilon_k += s.m * s.epsilon_k * n;
            mu += s.mu * n;
            q += s.q * n;
            z += s.z * n;
        });

        // entropy scaling
        let mut viscosity = if segments
            .iter()
            .all(|(record, _)| record.viscosity.is_some())
        {
            Some([0.0; 4])
        } else {
            None
        };
        let mut thermal_conductivity = if segments
            .iter()
            .all(|(record, _)| record.thermal_conductivity.is_some())
        {
            Some([0.0; 4])
        } else {
            None
        };
        let diffusion = if segments
            .iter()
            .all(|(record, _)| record.diffusion.is_some())
        {
            Some([0.0; 5])
        } else {
            None
        };

        let n_t = segments.iter().fold(0.0, |acc, (_, n)| acc + n);
        segments.iter().for_each(|(s, n)| {
            let s3 = s.m * s.sigma.powi(3) * n;
            if let Some(p) = viscosity.as_mut() {
                let [a, b, c, d] = s.viscosity.unwrap();
                p[0] += s3 * a;
                p[1] += s3 * b / sigma3.powf(0.45);
                p[2] += n * c;
                p[3] += n * d;
            }
            if let Some(p) = thermal_conductivity.as_mut() {
                let [a, b, c, d] = s.thermal_conductivity.unwrap();
                p[0] += n * a;
                p[1] += n * b;
                p[2] += n * c;
                p[3] += n_t * d;
            }
            // if let Some(p) = diffusion.as_mut() {
            //     let [a, b, c, d, e] = s.diffusion.unwrap();
            //     p[0] += s3 * a;
            //     p[1] += s3 * b / sigma3.powf(0.45);
            //     p[2] += *n * c;
            //     p[3] += *n * d;
            // }
        });
        // correction due to difference in Chapman-Enskog reference between GC and regular formulation.
        viscosity = viscosity.map(|v| [v[0] - 0.5 * m.ln(), v[1], v[2], v[3]]);

        Ok(Self {
            m,
            sigma: (sigma3 / m).cbrt(),
            epsilon_k: epsilon_k / m,
            mu,
            q,
            z,
            permittivity_record: None,
            viscosity,
            diffusion,
            thermal_conductivity,
        })
    }
}

impl EPcpSaftRecord {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: f64,
        q: f64,
        z: f64,
        permittivity_record: Option<PermittivityRecord>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> EPcpSaftRecord {
        EPcpSaftRecord {
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            z,
            permittivity_record,
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

/// Association parameters for the ePCP-SAFT equation of state.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct EPcpSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl EPcpSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

impl CombiningRule<EPcpSaftRecord> for EPcpSaftAssociationRecord {
    fn combining_rule(
        _: &EPcpSaftRecord,
        _: &EPcpSaftRecord,
        parameters_i: &Self,
        parameters_j: &Self,
    ) -> Self {
        let kappa_ab = (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt();
        let epsilon_k_ab = 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab);
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// ePCP-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct EPcpSaftBinaryRecord {
    /// Binary dispersion interaction parameter
    pub k_ij: f64,
}

impl EPcpSaftBinaryRecord {
    pub fn new(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

impl FromSegmentsBinary for EPcpSaftBinaryRecord {
    fn from_segments_binary(segments: &[(Self, f64, f64)]) -> FeosResult<Self> {
        let (k_ij, n) = segments.iter().fold((0.0, 0.0), |(k_ij, n), (br, n1, n2)| {
            let nab = n1 * n2;
            (k_ij + br.k_ij * nab, n + nab)
        });
        Ok(Self { k_ij: k_ij / n })
    }
}

/// Parameter set required for the ePCP-SAFT equation of state and Helmholtz energy functional.
pub type EPcpSaftParameters =
    Parameters<EPcpSaftRecord, EPcpSaftBinaryRecord, EPcpSaftAssociationRecord>;

/// The ePCP-SAFT parameters in an easier accessible format.
///
/// All model parameters are of the generic type `D` (`f64` or a dual number for derivatives
/// with respect to model parameters). `m` to `born_delta` are the adjustable parameters,
/// `mu2` to `e_k_ij` are derived from them, the remaining fields describe the structure.
#[derive(Clone, Debug)]
pub struct EPcpSaftPars<D = f64> {
    pub m: DVector<D>,
    pub sigma: DVector<D>,
    pub epsilon_k: DVector<D>,
    /// dipole moment in Debye
    pub mu: DVector<D>,
    /// quadrupole moment in Debye Angstrom
    pub q: DVector<D>,
    pub k_ij: DMatrix<D>,
    /// (dipole_scaling, polarizability_scaling, correlation_integral_parameter) of all components;
    /// `None` if at least one component has no permittivity record.
    pub permittivity: Option<[DVector<D>; 3]>,
    /// correction of the Born diameter of all cations and all anions in Angstrom
    pub born_delta: [D; 2],
    pub mu2: DVector<D>,
    pub q2: DVector<D>,
    pub sigma_ij: DMatrix<D>,
    pub epsilon_k_ij: DMatrix<D>,
    pub e_k_ij: DMatrix<D>,
    pub ndipole: usize,
    pub nquadpole: usize,
    pub dipole_comp: Vec<usize>,
    pub quadpole_comp: Vec<usize>,
    pub z: DVector<f64>,
    pub nionic: usize,
    pub ionic_comp: Vec<usize>,
    pub viscosity: Option<DMatrix<f64>>,
    pub diffusion: Option<DMatrix<f64>>,
    pub thermal_conductivity: Option<DMatrix<f64>>,
}

impl EPcpSaftPars {
    pub fn new(parameters: &EPcpSaftParameters) -> FeosResult<Self> {
        let n = parameters.pure.len();

        let [m, sigma, epsilon_k, mu, q, z] =
            parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k, pr.mu, pr.q, pr.z]);
        let [viscosity, thermal_conductivity] =
            parameters.collate(|pr| [pr.viscosity, pr.thermal_conductivity]);
        let [diffusion] = parameters.collate(|pr| [pr.diffusion]);
        let [k_ij] = parameters.collate_binary(|br| [br.k_ij]);

        // ions
        let ionic_comp: Vec<usize> = (0..n).filter(|&i| z[i] != 0.0).collect();
        let nionic = ionic_comp.len();

        // permittivity
        let permittivity = parameters
            .pure
            .iter()
            .map(|pr| pr.model_record.permittivity_record)
            .collect::<Option<Vec<_>>>()
            .map(|records| {
                let [mut a, mut b, mut c] =
                    [DVector::zeros(n), DVector::zeros(n), DVector::zeros(n)];
                for (i, r) in records.into_iter().enumerate() {
                    let PermittivityRecord::PerturbationTheory {
                        dipole_scaling,
                        polarizability_scaling,
                        correlation_integral_parameter,
                    } = r;
                    a[i] = dipole_scaling;
                    b[i] = polarizability_scaling;
                    c[i] = correlation_integral_parameter;
                }
                [a, b, c]
            });
        if nionic > 0 && permittivity.is_none() {
            return Err(FeosError::IncompatibleParameters(
                "ePCP-SAFT: provide a permittivity record for every component if ions are present."
                    .to_string(),
            ));
        }

        let dipole_comp: Vec<usize> = (0..n).filter(|&i| mu[i] != 0.0).collect();
        let ndipole = dipole_comp.len();
        let quadpole_comp: Vec<usize> = (0..n).filter(|&i| q[i] != 0.0).collect();
        let nquadpole = quadpole_comp.len();

        let collect = |coefficients: Vec<Option<Vec<f64>>>, rows: usize| {
            if coefficients.iter().any(|v| v.is_none()) {
                None
            } else {
                let mut v = DMatrix::zeros(rows, coefficients.len());
                for (i, vi) in coefficients.into_iter().enumerate() {
                    v.column_mut(i).add_assign(&DVector::from(vi.unwrap()));
                }
                Some(v)
            }
        };
        let viscosity = collect(viscosity.iter().map(|v| v.map(|v| v.to_vec())).collect(), 4);
        let diffusion = collect(diffusion.iter().map(|v| v.map(|v| v.to_vec())).collect(), 5);
        let thermal_conductivity = collect(
            thermal_conductivity
                .iter()
                .map(|v| v.map(|v| v.to_vec()))
                .collect(),
            4,
        );

        Ok(Self::derive(
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            k_ij,
            permittivity,
            [0.0, 0.0],
            Self {
                m: DVector::zeros(0),
                sigma: DVector::zeros(0),
                epsilon_k: DVector::zeros(0),
                mu: DVector::zeros(0),
                q: DVector::zeros(0),
                k_ij: DMatrix::zeros(0, 0),
                permittivity: None,
                born_delta: [0.0; 2],
                mu2: DVector::zeros(0),
                q2: DVector::zeros(0),
                sigma_ij: DMatrix::zeros(0, 0),
                epsilon_k_ij: DMatrix::zeros(0, 0),
                e_k_ij: DMatrix::zeros(0, 0),
                ndipole,
                nquadpole,
                dipole_comp,
                quadpole_comp,
                z,
                nionic,
                ionic_comp,
                viscosity,
                diffusion,
                thermal_conductivity,
            },
        ))
    }
}

impl<D: DualNum<f64> + Copy> EPcpSaftPars<D> {
    /// Parameters from the adjustable parameters; the structure is taken from `structure`.
    #[expect(clippy::too_many_arguments)]
    fn derive<D0>(
        m: DVector<D>,
        sigma: DVector<D>,
        epsilon_k: DVector<D>,
        mu: DVector<D>,
        q: DVector<D>,
        mut k_ij: DMatrix<D>,
        permittivity: Option<[DVector<D>; 3]>,
        born_delta: [D; 2],
        structure: EPcpSaftPars<D0>,
    ) -> Self {
        let n = m.len();
        let z = &structure.z;

        // no dispersion between ions of the same charge sign (Held et al. 2014)
        for &i in &structure.ionic_comp {
            for &j in &structure.ionic_comp {
                if z[i] * z[j] > 0.0 {
                    k_ij[(i, j)] = D::one();
                }
            }
        }

        // 1 D^2 = 1e-19 J A^3
        let factor = (JOULE / KELVIN / KB).into_value();
        let mu2 = DVector::from_fn(n, |i, _| {
            mu[i] * mu[i] / (m[i] * sigma[i].powi(3) * epsilon_k[i]) * 1e-19 * factor
        });
        let q2 = DVector::from_fn(n, |i, _| {
            q[i] * q[i] / (m[i] * sigma[i].powi(5) * epsilon_k[i]) * 1e-19 * factor
        });

        // sqrt(0) of a dual number has infinite derivatives (NaN after multiplication with 0)
        let e_k_ij = DMatrix::from_fn(n, n, |i, j| {
            let e2 = epsilon_k[i] * epsilon_k[j];
            if e2.re() == 0.0 { D::zero() } else { e2.sqrt() }
        });
        let sigma_ij = DMatrix::from_fn(n, n, |i, j| (sigma[i] + sigma[j]) * 0.5);
        let epsilon_k_ij = DMatrix::from_fn(n, n, |i, j| (-k_ij[(i, j)] + 1.0) * e_k_ij[(i, j)]);

        Self {
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            k_ij,
            permittivity,
            born_delta,
            mu2,
            q2,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            ndipole: structure.ndipole,
            nquadpole: structure.nquadpole,
            dipole_comp: structure.dipole_comp,
            quadpole_comp: structure.quadpole_comp,
            z: structure.z,
            nionic: structure.nionic,
            ionic_comp: structure.ionic_comp,
            viscosity: structure.viscosity,
            diffusion: structure.diffusion,
            thermal_conductivity: structure.thermal_conductivity,
        }
    }

    /// Parameters with all adjustable parameters mapped by `f`; derived parameters are recomputed.
    pub fn map<D2: DualNum<f64> + Copy>(&self, f: impl Fn(D) -> D2) -> EPcpSaftPars<D2> {
        let v = |x: &DVector<D>| x.map(&f);
        EPcpSaftPars::derive(
            v(&self.m),
            v(&self.sigma),
            v(&self.epsilon_k),
            v(&self.mu),
            v(&self.q),
            self.k_ij.map(&f),
            self.permittivity.as_ref().map(|p| p.each_ref().map(v)),
            self.born_delta.map(&f),
            self.clone(),
        )
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use crate::epcpsaft::EPcpSaft;
    use feos_core::parameter::{BinarySegmentRecord, ChemicalRecord, PureRecord, SegmentRecord};

    pub fn propane_parameters() -> EPcpSaft {
        let propane_json = r#"
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "m": 2.001829,
                "sigma": 3.618353,
                "epsilon_k": 208.1101,
                "viscosity": [-0.8013, -1.9972,-0.2907, -0.0467],
                "thermal_conductivity": [-0.15348,  -0.6388, 1.21342, -0.01664],
                "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0],
                "molarweight": 44.0962
            }"#;
        let propane_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(propane_json).expect("Unable to parse json.");
        EPcpSaft::new(EPcpSaftParameters::new_pure(propane_record).unwrap()).unwrap()
    }

    pub fn carbon_dioxide_parameters() -> EPcpSaftPars {
        let co2_json = r#"
        {
            "identifier": {
                "cas": "124-38-9",
                "name": "carbon-dioxide",
                "iupac_name": "carbon dioxide",
                "smiles": "O=C=O",
                "inchi": "InChI=1/CO2/c2-1-3",
                "formula": "CO2"
            },
            "molarweight": 44.0098,
            "m": 1.5131,
            "sigma": 3.1869,
            "epsilon_k": 163.333,
            "q": 4.4
        }"#;
        let co2_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(co2_json).expect("Unable to parse json.");
        EPcpSaftPars::new(&EPcpSaftParameters::new_pure(co2_record).unwrap()).unwrap()
    }

    pub fn butane_parameters() -> EPcpSaft {
        let butane_json = r#"
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "m": 2.331586,
                "sigma": 3.7086010000000003,
                "epsilon_k": 222.8774,
                "molarweight": 58.123
            }"#;
        let butane_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(butane_json).expect("Unable to parse json.");
        EPcpSaft::new(EPcpSaftParameters::new_pure(butane_record).unwrap()).unwrap()
    }

    pub fn nonane_parameters() -> EPcpSaft {
        let nonane_json = r#"
            {
                "identifier": {
                    "cas": "111-84-2",
                    "name": "nonane",
                    "iupac_name": "nonane",
                    "smiles": "CCCCCCCCC",
                    "inchi": "InChI=1S/C9H20/c1-3-5-7-9-8-6-4-2/h3-9H2,1-2H3",
                    "formula": "C9H20"
                },
                "molarweight": 128.258,
                "m": 4.2079,
                "sigma": 3.8448,
                "epsilon_k": 244.51,
                "viscosity": [
                    -1.4629,
                    -3.0058,
                    -0.5842,
                    -0.1222
                ]
            }"#;
        let nonane_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(nonane_json).expect("Unable to parse json.");
        EPcpSaft::new(EPcpSaftParameters::new_pure(nonane_record).unwrap()).unwrap()
    }

    pub fn dme_parameters() -> EPcpSaftPars {
        let dme_json = r#"
            {
                "identifier": {
                    "cas": "115-10-6",
                    "name": "dimethyl-ether",
                    "iupac_name": "methoxymethane",
                    "smiles": "COC",
                    "inchi": "InChI=1/C2H6O/c1-3-2/h1-2H3",
                    "formula": "C2H6O"
                },
                "m": 2.2634,
                "sigma": 3.2723,
                "epsilon_k": 210.29,
                "mu": 1.3,
                "molarweight": 46.0688
            }"#;
        let dme_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(dme_json).expect("Unable to parse json.");
        EPcpSaftPars::new(&EPcpSaftParameters::new_pure(dme_record).unwrap()).unwrap()
    }

    #[expect(dead_code)]
    pub fn water_parameters(na: f64) -> EPcpSaftParameters {
        let water_json = r#"
            {
                "identifier": {
                    "cas": "7732-18-5",
                    "name": "water_np",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "molarweight": 18.0152,
                "association_sites": [
                    {
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let mut water_record: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(water_json).expect("Unable to parse json.");
        water_record.association_sites[0].na = na;
        EPcpSaftParameters::new_pure(water_record).unwrap()
    }

    pub fn dme_co2_parameters() -> EPcpSaftPars {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "115-10-6",
                    "name": "dimethyl-ether",
                    "iupac_name": "methoxymethane",
                    "smiles": "COC",
                    "inchi": "InChI=1/C2H6O/c1-3-2/h1-2H3",
                    "formula": "C2H6O"
                },
                "molarweight": 46.0688,
                "m": 2.2634,
                "sigma": 3.2723,
                "epsilon_k": 210.29,
                "mu": 1.3
            },
            {
                "identifier": {
                    "cas": "124-38-9",
                    "name": "carbon-dioxide",
                    "iupac_name": "carbon dioxide",
                    "smiles": "O=C=O",
                    "inchi": "InChI=1/CO2/c2-1-3",
                    "formula": "CO2"
                },
                "molarweight": 44.0098,
                "m": 1.5131,
                "sigma": 3.1869,
                "epsilon_k": 163.333,
                "q": 4.4
            }
        ]"#;
        let binary_record: [PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        EPcpSaftPars::new(&EPcpSaftParameters::new_binary(binary_record, None, vec![]).unwrap())
            .unwrap()
    }

    pub fn propane_butane_parameters() -> EPcpSaft {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "m": 2.0018290000000003,
                "sigma": 3.618353,
                "epsilon_k": 208.1101,
                "viscosity": [-0.8013, -1.9972, -0.2907, -0.0467],
                "thermal_conductivity": [-0.15348, -0.6388, 1.21342, -0.01664],
                "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0],
                "molarweight": 44.0962
            },
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "m": 2.331586,
                "sigma": 3.7086010000000003,
                "epsilon_k": 222.8774,
                "viscosity": [-0.9763, -2.2413, -0.3690, -0.0605],
                "diffusion": [-0.8985872992958458, 0.3428584416613513, 0.10236616087103916, 0.0, 0.0],
                "molarweight": 58.123
            }
        ]"#;
        let binary_record: [PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        EPcpSaft::new(EPcpSaftParameters::new_binary(binary_record, None, vec![]).unwrap()).unwrap()
    }

    pub fn nonane_heptane_parameters() -> EPcpSaft {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "111-84-2",
                    "name": "nonane",
                    "iupac_name": "nonane",
                    "smiles": "CCCCCCCCC",
                    "inchi": "InChI=1S/C9H20/c1-3-5-7-9-8-6-4-2/h3-9H2,1-2H3",
                    "formula": "C9H20"
                },
                "molarweight": 128.258,
                "m": 4.2079,
                "sigma": 3.8448,
                "epsilon_k": 244.51,
                "viscosity": [
                    -1.4629,
                    -3.0058,
                    -0.5842,
                    -0.1222
                ]
            },
            {
                "identifier": {
                    "cas": "142-82-5",
                    "name": "heptane",
                    "iupac_name": "heptane",
                    "smiles": "CCCCCCC",
                    "inchi": "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3",
                    "formula": "C7H16"
                },
                "molarweight": 100.204,
                "m": 3.4831,
                "sigma": 3.8049,
                "epsilon_k": 238.4,
                "viscosity": [
                    -1.2979,
                    -2.6936,
                    -0.4951,
                    -0.0988
                ]
            }
        ]"#;
        let binary_record: [PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        EPcpSaft::new(EPcpSaftParameters::new_binary(binary_record, None, vec![]).unwrap()).unwrap()
    }

    /// Water (Esper et al. 2023 with the permittivity parameters of
    /// parameters/epcpsaft/esper2023.json), Na+ and Cl- (sigma, epsilon_k of Held et al. 2014;
    /// permittivity parameters of the ions are arbitrary test values).
    pub fn water_nacl_parameters() -> EPcpSaftParameters {
        EPcpSaftParameters::new(water_nacl_records(), vec![]).unwrap()
    }

    pub fn water_nacl_records() -> Vec<PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord>> {
        let json = r#"[
            {
                "identifier": {"name": "water", "cas": "7732-18-5"},
                "molarweight": 18.011,
                "m": 2.36948,
                "sigma": 2.15072,
                "epsilon_k": 230.71557,
                "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.35319, "epsilon_k_ab": 2195.10176}],
                "permittivity_record": {"PerturbationTheory": {"dipole_scaling": 5.2607, "polarizability_scaling": 0.0, "correlation_integral_parameter": 0.1320}}
            },
            {
                "identifier": {"name": "sodium ion"},
                "molarweight": 22.98977,
                "m": 1.0,
                "sigma": 2.8232,
                "epsilon_k": 230.0,
                "z": 1.0,
                "permittivity_record": {"PerturbationTheory": {"dipole_scaling": 4.0, "polarizability_scaling": 0.0, "correlation_integral_parameter": 0.1}}
            },
            {
                "identifier": {"name": "chloride ion"},
                "molarweight": 35.453,
                "m": 1.0,
                "sigma": 2.7560,
                "epsilon_k": 170.0,
                "z": -1.0,
                "permittivity_record": {"PerturbationTheory": {"dipole_scaling": 8.0, "polarizability_scaling": 0.0, "correlation_integral_parameter": 0.2}}
            }
        ]"#;
        serde_json::from_str(json).expect("Unable to parse json.")
    }

    #[test]
    pub fn test_kij() -> FeosResult<()> {
        let ch3: String = "CH3".into();
        let ch2: String = "CH2".into();
        let oh = "OH".into();
        let propane = ChemicalRecord::new(
            Default::default(),
            vec![ch3.clone(), ch2.clone(), ch3.clone()],
            None,
        );
        let ethanol = ChemicalRecord::new(Default::default(), vec![ch3, ch2, oh], None);
        let segment_records =
            SegmentRecord::from_json("../../parameters/pcsaft/sauer2014_homo.json")?;
        let kij = [("CH3", "OH", -0.2), ("CH2", "OH", -0.1)];
        let binary_segment_records: Vec<_> = kij
            .iter()
            .map(|&(id1, id2, k_ij)| {
                BinarySegmentRecord::new(
                    id1.into(),
                    id2.into(),
                    Some(EPcpSaftBinaryRecord { k_ij }),
                )
            })
            .collect();
        let params = EPcpSaftParameters::from_segments(
            vec![propane, ethanol],
            &segment_records,
            Some(&binary_segment_records),
        )?;
        assert_eq!(params.binary[0].id1, 0);
        assert_eq!(params.binary[0].id2, 1);
        assert_eq!(params.binary[0].model_record.k_ij, -0.5 / 9.);

        Ok(())
    }

    #[test]
    fn test_association_json() -> FeosResult<()> {
        let json1 = r#"
            {
                "identifier": {
                    "name": "comp1"
                },
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "association_sites": [
                    {
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let record1: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(json1)?;

        let json2 = r#"
            {
                "identifier": {
                    "name": "comp2"
                },
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "association_sites": [
                    {
                        "id": "site1",
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    },
                    {
                        "id": "site2",
                        "kappa_ab": 0.01,
                        "epsilon_k_ab": 2200.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let record2: PureRecord<EPcpSaftRecord, EPcpSaftAssociationRecord> =
            serde_json::from_str(json2)?;

        // println!("{record1}");
        // println!("{record2}");

        assert_eq!(
            record1.association_sites[0].parameters.unwrap().kappa_ab,
            record2.association_sites[0].parameters.unwrap().kappa_ab
        );

        Ok(())
    }
}
