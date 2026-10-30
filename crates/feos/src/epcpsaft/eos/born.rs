use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::DualNum;

/// Born contribution (Bülow et al. 2021, Rueben et al. 2024).
pub struct Born;

impl Born {
    /// Reduced Helmholtz energy density
    /// a / (kT V) = -lambda_B (eps_r - 1) sum_i rho_i z_i^2 / d_Born,i
    /// with lambda_B = e^2 / (4 pi eps_0 eps_r kB T), i.e. -e^2 / (4 pi eps_0 kB T) (1 - 1/eps_r) sum_i rho_i z_i^2 / d_Born,i.
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &EPcpSaftPars,
        state: &StateHD<D>,
        born_diameter: &DVector<D>,
        bjerrum_length: D,
        permittivity: D,
    ) -> D {
        let p = parameters;
        let mut sum_rho_z2_d = D::zero();
        for &i in &p.ionic_comp {
            sum_rho_z2_d += state.partial_density[i] * p.z[i].powi(2) / born_diameter[i];
        }
        -bjerrum_length * (permittivity - 1.0) * sum_rho_z2_d
    }
}
