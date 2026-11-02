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
        parameters: &EPcpSaftPars<D>,
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
            let chi = chi(kappa * diameter[i]);
            sum_rho_z2_chi += state.partial_density[i] * p.z[i].powi(2) * chi;
        }

        -kappa * bjerrum_length * sum_rho_z2_chi / 3.0
    }
}

/// Below this value of kappa d, chi is evaluated from its Taylor series.
const CHI_SERIES_LIMIT: f64 = 0.5;
/// Number of terms of the Taylor series (truncation error 3 0.5^52 / 55 < 1e-16).
const CHI_SERIES_TERMS: usize = 52;

/// chi(x) = 3 / x^3 [ln(1 + x) - 2 (1 + x) + (1 + x)^2 / 2 + 3 / 2].
///
/// The bracket equals ln(1 + x) - x + x^2 / 2 = x^3 / 3 - x^4 / 4 + ... and is the difference of
/// terms of order 1, so its relative rounding error grows as 1 / x^3. For x < 0.5 the series
/// chi(x) = sum_k 3 (-x)^k / (k + 3) is used instead.
fn chi<D: DualNum<f64> + Copy>(x: D) -> D {
    if x.re() < CHI_SERIES_LIMIT {
        chi_series(x)
    } else {
        chi_closed_form(x)
    }
}

fn chi_series<D: DualNum<f64> + Copy>(x: D) -> D {
    (0..CHI_SERIES_TERMS)
        .rev()
        .fold(D::zero(), |s, k| -s * x + 3.0 / (k + 3) as f64)
}

fn chi_closed_form<D: DualNum<f64> + Copy>(x: D) -> D {
    x.powi(3).recip() * ((x + 1.0).ln() - (x + 1.0) * 2.0 + (x + 1.0).powi(2) * 0.5 + 1.5) * 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_dual::Dual2_64;

    /// Series and closed form agree (value, first and second derivative) where both are accurate.
    #[test]
    fn chi_series_closed_form() {
        for x in [0.3, 0.45, CHI_SERIES_LIMIT, 0.55] {
            let x = Dual2_64::from_re(x).derivative();
            let (s, c) = (chi_series(x), chi_closed_form(x));
            assert_relative_eq!(s.re, c.re, max_relative = 1e-13);
            assert_relative_eq!(s.v1, c.v1, max_relative = 1e-11);
            assert_relative_eq!(s.v2, c.v2, max_relative = 1e-9);
        }
    }

    /// chi = 1 - 3 x / 4 + 3 x^2 / 5 - x^3 / 2 + 3 x^4 / 7 + O(x^5) for x -> 0.
    #[test]
    fn chi_small_argument() {
        for x in [1e-3, 1e-6, 1e-12] {
            let series = 1.0 - 0.75 * x + 0.6 * x.powi(2) - 0.5 * x.powi(3) + 3.0 / 7.0 * x.powi(4);
            assert_relative_eq!(chi(x), series, max_relative = 1e-15);
        }
    }
}
