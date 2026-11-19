use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::{Dual, DualNum};
use std::f64::consts::PI;

/// Debye-Hückel contribution (Cameretti et al. 2005, Held et al. 2014).
pub struct Ionic;

impl Ionic {
    /// Reduced Helmholtz energy density
    /// a / (kT V) = -kappa lambda_B / 3 sum_i rho_i z_i^2 chi_i,
    /// kappa^2 = 4 pi lambda_B sum_i rho_i z_i^2,
    /// chi_i = 3 / (kappa d_i)^3 [ln(1 + kappa d_i) - 2 (1 + kappa d_i) + (1 + kappa d_i)^2 / 2 + 3 / 2].
    pub fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
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

/// Reduced Helmholtz energy density of the Debye-Hückel term for ions with densities `rho` (Å⁻³), charges `z`,
/// diameters `d` (Å) and Bjerrum length `lb` (Å):
/// a / (kT V) = -κ λ_B / 3 Σ_i ρ_i z_i² χ(κ d_i), κ² = 4π λ_B Σ_i ρ_i z_i².
pub fn debye_hueckel<D: DualNum<Primitive = f64> + Copy>(
    rho: &[D],
    z: &[f64],
    d: &[D],
    lb: D,
) -> D {
    let sum_rho_z2 = rho
        .iter()
        .zip(z)
        .fold(D::zero(), |acc, (&r, &z)| acc + r * z * z);
    if sum_rho_z2.re() == 0.0 {
        return D::zero();
    }
    let kappa = (lb * sum_rho_z2 * 4.0 * PI).sqrt();
    let sum = rho
        .iter()
        .zip(z)
        .zip(d)
        .fold(D::zero(), |acc, ((&r, &z), &d)| {
            acc + r * z * z * chi(kappa * d)
        });
    -kappa * lb * sum / 3.0
}

/// Reduced chemical potential β μ_k = ∂(a / kT V)/∂ρ_k of a species with charge `q` and diameter `sigma` (Å) in the
/// Debye-Hückel term of the ions `rho`, `z`, `d` at constant Bjerrum length `lb` (the species may be one of the ions
/// or a test particle at zero density):
/// β μ_k = -λ_B κ q² χ(κ σ) / 3 - 2π λ_B² q² / (3 κ) Σ_i ρ_i z_i² [χ(κ d_i) + κ d_i χ'(κ d_i)],
/// with the limit -λ_B κ q² / 2 for κ d -> 0 (Debye-Hückel limiting law).
pub fn debye_hueckel_chemical_potential<D: DualNum<Primitive = f64> + Copy>(
    q: f64,
    sigma: D,
    rho: &[D],
    z: &[f64],
    d: &[D],
    lb: D,
) -> D {
    let sum_rho_z2 = rho
        .iter()
        .zip(z)
        .fold(D::zero(), |acc, (&r, &z)| acc + r * z * z);
    if sum_rho_z2.re() == 0.0 || q == 0.0 {
        return D::zero();
    }
    let kappa = (lb * sum_rho_z2 * 4.0 * PI).sqrt();
    let mut sum = D::zero();
    for ((&r, &z), &d) in rho.iter().zip(z).zip(d) {
        // χ(x) + x χ'(x) = d(x χ(x))/dx
        let x = Dual::new(kappa * d, D::one());
        let xchi = x * chi(x);
        sum += r * z * z * xchi.eps;
    }
    -lb * kappa * q * q * chi(kappa * sigma) / 3.0
        - lb * lb * q * q * sum * (2.0 * PI / 3.0) / kappa
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
fn chi<D: DualNum<Primitive = f64> + Copy>(x: D) -> D {
    if x.re() < CHI_SERIES_LIMIT {
        chi_series(x)
    } else {
        chi_closed_form(x)
    }
}

fn chi_series<D: DualNum<Primitive = f64> + Copy>(x: D) -> D {
    (0..CHI_SERIES_TERMS)
        .rev()
        .fold(D::zero(), |s, k| -s * x + 3.0 / (k + 3) as f64)
}

fn chi_closed_form<D: DualNum<Primitive = f64> + Copy>(x: D) -> D {
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

    /// The chemical potential equals the derivative of the Helmholtz energy density with respect to the density of
    /// the species (for an ion of the mixture and for a test particle at zero density), and tends to the limiting
    /// law at low concentration.
    #[test]
    fn chemical_potential() {
        use num_dual::Dual64;
        let lb = 7.1;
        let z = [1.0, -1.0, 2.0, -2.0];
        let d = [2.5, 3.1, 2.8, 3.6];
        let rho = [0.004, 0.006, 0.002, 0.001];
        for k in 0..4 {
            let rho_d: Vec<Dual64> = (0..4)
                .map(|i| Dual64::new(rho[i], if i == k { 1.0 } else { 0.0 }))
                .collect();
            let d_d: Vec<Dual64> = d.iter().map(|&x| Dual64::from(x)).collect();
            let mu_ad = debye_hueckel(&rho_d, &z, &d_d, Dual64::from(lb)).eps;
            let mu = debye_hueckel_chemical_potential(z[k], d[k], &rho, &z, &d, lb);
            assert_relative_eq!(mu, mu_ad, max_relative = 1e-12);
        }
        // test particle of charge 3 and diameter 4 at zero density
        let mut rho5: Vec<Dual64> = rho.iter().map(|&r| Dual64::from(r)).collect();
        rho5.push(Dual64::new(0.0, 1.0));
        let mut z5 = z.to_vec();
        z5.push(3.0);
        let mut d5: Vec<Dual64> = d.iter().map(|&x| Dual64::from(x)).collect();
        d5.push(Dual64::from(4.0));
        let mu_ad = debye_hueckel(&rho5, &z5, &d5, Dual64::from(lb)).eps;
        assert_relative_eq!(
            debye_hueckel_chemical_potential(3.0, 4.0, &rho, &z, &d, lb),
            mu_ad,
            max_relative = 1e-12
        );
        // limiting law
        let rho_low: Vec<f64> = rho.iter().map(|r| r * 1e-10).collect();
        let kappa =
            (4.0 * PI * lb * rho_low.iter().zip(&z).map(|(r, z)| r * z * z).sum::<f64>()).sqrt();
        assert_relative_eq!(
            debye_hueckel_chemical_potential(2.0, 3.0, &rho_low, &z, &d, lb),
            -lb * kappa * 4.0 / 2.0,
            max_relative = 1e-4
        );
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
