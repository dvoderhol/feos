use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::DualNum;
use std::f64::consts::PI;

/// Debye-Hückel contribution (Cameretti et al. 2005, Held et al. 2014).
pub struct Ionic;

impl Ionic {
    /// Reduced Helmholtz energy density
    /// a / (kT V) = -kappa lambda_B / 3 sum_i rho_i z_i^2 chi_i,
    /// kappa^2 = 4 pi lambda_B sum_i rho_i z_i^2,
    /// chi_i = 3 / (kappa d_i)^3 [ln(1 + kappa d_i) - 2 (1 + kappa d_i) + (1 + kappa d_i)^2 / 2 + 3 / 2].
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &EPcpSaftPars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        bjerrum_length: D,
    ) -> D {
        let p = parameters;

        let mut sum_rho_z2 = D::zero();
        for &i in &p.ionic_comp {
            sum_rho_z2 += state.partial_density[i] * p.z[i].powi(2);
        }
        if sum_rho_z2.re() == 0.0 {
            return D::zero();
        }

        let kappa = (bjerrum_length * sum_rho_z2 * 4.0 * PI).sqrt();

        let mut sum_rho_z2_chi = D::zero();
        for &i in &p.ionic_comp {
            let kd = kappa * diameter[i];
            let chi = kd.powi(3).recip()
                * ((kd + 1.0).ln() - (kd + 1.0) * 2.0 + (kd + 1.0).powi(2) * 0.5 + 1.5)
                * 3.0;
            sum_rho_z2_chi += state.partial_density[i] * p.z[i].powi(2) * chi;
        }

        -kappa * bjerrum_length * sum_rho_z2_chi / 3.0
    }
}
