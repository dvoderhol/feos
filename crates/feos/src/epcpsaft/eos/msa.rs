use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::{Dual, DualNum};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_6, PI};

/// Mean spherical approximation (MSA) of the primitive model of electrolytes with ions of
/// different diameters (Blum 1975; equations as in Maribo-Mogensen et al. 2012, eqs. 25-28 and 32).
///
/// a / (kT V) = Γ³ / (3π) - λ_B Σ_i ρ_i z_i (Γ z_i + η d_i) / (1 + Γ d_i),
/// with the screening parameter Γ from
/// Γ² = π λ_B Σ_k ρ_k ((z_k - η d_k²) / (1 + Γ d_k))²,
/// η = π / (2 Δ Ω) Σ_k ρ_k d_k z_k / (1 + Γ d_k), Ω = 1 + π / (2 Δ) Σ_k ρ_k d_k³ / (1 + Γ d_k),
/// Δ = 1 - π/6 Σ_k ρ_k d_k³. All sums run over the ions (primitive model: the solvent is a
/// dielectric continuum), except Δ with [MsaPacking::Total]: Δ = 1 - π/6 Σ_k ρ_k m_k d_k³ over all
/// species (excluded volume of the solvent included, as in Maribo-Mogensen et al. 2012, eq. 28).
/// With [MsaPacking::Total], Ω and η keep the sums over the ions; in Blum's solution for ions and
/// neutral hard spheres, the neutral species would also enter Ω and the equation for Γ.
pub struct Msa(pub MsaPacking);

/// Species in the packing term Δ of the MSA.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MsaPacking {
    /// Ions only (primitive model; Blum 1975).
    Ions,
    /// All species (segments of the solvent included; Maribo-Mogensen et al. 2012).
    Total,
}

/// Newton iterations after the real part of Γ converged; each iteration doubles the order of
/// derivatives that are exact (nested dual numbers up to order 15).
const EXTRA_ITERATIONS: usize = 4;
const MAX_ITER: usize = 100;
const TOL: f64 = 1e-15;

impl Msa {
    /// Reduced Helmholtz energy density with the screening parameter solved for the given
    /// densities, diameters `diameter` and Bjerrum length `bjerrum_length`.
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &EPcpSaftPars<D>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        bjerrum_length: D,
    ) -> D {
        let p = parameters;
        let rho: Vec<D> = p
            .ionic_comp
            .iter()
            .map(|&i| state.partial_density[i])
            .collect();
        let z: Vec<f64> = p.ionic_comp.iter().map(|&i| p.z[i]).collect();
        let d: Vec<D> = p.ionic_comp.iter().map(|&i| diameter[i]).collect();

        let sum_rho_z2 = rho
            .iter()
            .zip(&z)
            .fold(D::zero(), |acc, (&r, &z)| acc + r * z * z);
        if sum_rho_z2.re() == 0.0 {
            return D::zero();
        }
        let delta = match self.0 {
            MsaPacking::Ions => {
                -rho.iter()
                    .zip(&d)
                    .fold(D::zero(), |acc, (&r, &d)| acc + r * d.powi(3))
                    * FRAC_PI_6
                    + 1.0
            }
            MsaPacking::Total => {
                -(0..p.m.len()).fold(D::zero(), |acc, i| {
                    acc + state.partial_density[i] * p.m[i] * diameter[i].powi(3)
                }) * FRAC_PI_6
                    + 1.0
            }
        };

        let gamma = screening_parameter(&rho, &z, &d, bjerrum_length, delta, sum_rho_z2);
        let eta = eta(gamma, &rho, &z, &d, delta);
        let mut sum = D::zero();
        for ((&r, &z), &d) in rho.iter().zip(&z).zip(&d) {
            sum += r * z * (gamma * z + eta * d) / (gamma * d + 1.0);
        }
        gamma.powi(3) / (3.0 * PI) - bjerrum_length * sum
    }
}

/// η of the MSA (Maribo-Mogensen et al. 2012, eqs. 26-27)
fn eta<T: DualNum<f64> + Copy>(gamma: T, rho: &[T], z: &[f64], d: &[T], delta: T) -> T {
    let mut sum_z = T::zero();
    let mut sum_d3 = T::zero();
    for ((&r, &z), &d) in rho.iter().zip(z).zip(d) {
        let f = (gamma * d + 1.0).recip();
        sum_z += r * d * z * f;
        sum_d3 += r * d.powi(3) * f;
    }
    let omega = sum_d3 * FRAC_PI_2 / delta + 1.0;
    sum_z * FRAC_PI_2 / (delta * omega)
}

/// Γ(Γ) - Γ, with Γ(Γ) the right-hand side of the closure Γ² = π λ_B Σ ρ_k ((z_k - η d_k²)/(1 + Γ d_k))²
fn closure_residual<T: DualNum<f64> + Copy>(
    gamma: T,
    rho: &[T],
    z: &[f64],
    d: &[T],
    bjerrum_length: T,
    delta: T,
) -> T {
    let eta = eta(gamma, rho, z, d, delta);
    let mut sum = T::zero();
    for ((&r, &z), &d) in rho.iter().zip(z).zip(d) {
        sum += r * ((-eta * d * d + z) / (gamma * d + 1.0)).powi(2);
    }
    (bjerrum_length * sum * PI).sqrt() - gamma
}

/// Screening parameter Γ: Newton iteration in the number type D (exact derivatives with
/// respect to everything that D carries), started from the solution for η = 0 and equal
/// diameters, safeguarded by the bracket 0 < Γ <= κ/2.
fn screening_parameter<D: DualNum<f64> + Copy>(
    rho: &[D],
    z: &[f64],
    d: &[D],
    bjerrum_length: D,
    delta: D,
    sum_rho_z2: D,
) -> D {
    let kappa = (bjerrum_length * sum_rho_z2 * 4.0 * PI).sqrt();
    let d_mean = rho
        .iter()
        .zip(z)
        .zip(d)
        .fold(D::zero(), |acc, ((&r, &z), &d)| acc + r * z * z * d)
        / sum_rho_z2;
    let mut gamma = kappa / ((kappa * d_mean * 2.0 + 1.0).sqrt() + 1.0);
    let upper = kappa.re() * 0.5;

    let lift = |x: D| Dual::<D, f64>::from_re(x);
    let rho_l: Vec<_> = rho.iter().map(|&x| lift(x)).collect();
    let d_l: Vec<_> = d.iter().map(|&x| lift(x)).collect();
    let (lambda_l, delta_l) = (lift(bjerrum_length), lift(delta));

    let mut extra = 0;
    for _ in 0..MAX_ITER {
        let g = closure_residual(
            Dual::new(gamma, D::one()),
            &rho_l,
            z,
            &d_l,
            lambda_l,
            delta_l,
        );
        let mut next = gamma - g.re / g.eps;
        if !(next.re() > 0.0 && next.re() <= upper) {
            // safeguard: bisection towards the bracket (the residual decreases monotonically)
            let bound = if next.re() <= 0.0 { 0.0 } else { upper };
            next = (gamma + bound) * 0.5;
        }
        let converged = (next.re() - gamma.re()).abs() <= TOL * gamma.re();
        gamma = next;
        if converged {
            extra += 1;
            if extra > EXTRA_ITERATIONS {
                break;
            }
        }
    }
    gamma
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_dual::Dual64;

    /// restricted primitive model (equal diameters, symmetric salt): η = 0 and the closed-form
    /// solution Γ = (sqrt(1 + 2κd) - 1) / (2d) (Waisman and Lebowitz)
    #[test]
    fn restricted_primitive_model() {
        let (rho, z, d, lambda) = ([1e-3, 1e-3], [1.0, -1.0], [3.0, 3.0], 7.0);
        let delta = 1.0 - FRAC_PI_6 * 2e-3 * 27.0;
        let s = 2e-3;
        let gamma = screening_parameter(&rho, &z, &d, lambda, delta, s);
        let kappa = (4.0 * PI * lambda * s).sqrt();
        assert_relative_eq!(
            gamma,
            ((1.0 + 2.0 * kappa * 3.0).sqrt() - 1.0) / 6.0,
            max_relative = 1e-14
        );
        assert!(eta(gamma, &rho, &z, &d, delta).abs() < 1e-16);
    }

    /// unequal diameters and charges: Γ solves the closure; the Helmholtz energy is stationary
    /// with respect to Γ (Maribo-Mogensen et al. 2012, eq. 31)
    #[test]
    fn stationarity() {
        let (rho, z, d, lambda) = ([2e-3, 1e-3], [1.0, -2.0], [2.4, 3.6], 7.0);
        let delta = 1.0 - FRAC_PI_6 * (2e-3 * 2.4f64.powi(3) + 1e-3 * 3.6f64.powi(3));
        let s = 2e-3 + 4e-3;
        let gamma = screening_parameter(&rho, &z, &d, lambda, delta, s);
        assert!(closure_residual(gamma, &rho, &z, &d, lambda, delta).abs() < 1e-15);
        let a = |g: Dual64| {
            let (rho, d) = (rho.map(Dual64::from), d.map(Dual64::from));
            let eta = eta(g, &rho, &z, &d, Dual64::from(delta));
            let mut sum = Dual64::from(0.0);
            for i in 0..2 {
                sum += rho[i] * z[i] * (g * z[i] + eta * d[i]) / (g * d[i] + 1.0);
            }
            g.powi(3) / (3.0 * PI) - sum * lambda
        };
        let da = a(Dual64::from(gamma).derivative()).eps;
        assert!(
            da.abs() < 1e-14 * a(Dual64::from(gamma)).re.abs() / gamma,
            "{da}"
        );
    }

    /// derivatives of Γ with respect to the input (here the Bjerrum length) by implicit
    /// differentiation agree with a central difference
    #[test]
    fn screening_parameter_derivative() {
        let (rho, z, d) = ([2e-3, 1e-3], [1.0, -2.0], [2.4, 3.6]);
        let delta = 1.0 - FRAC_PI_6 * (2e-3 * 2.4f64.powi(3) + 1e-3 * 3.6f64.powi(3));
        let g = |lambda: Dual64| {
            let rho = rho.map(Dual64::from);
            let d = d.map(Dual64::from);
            screening_parameter(
                &rho,
                &z,
                &d,
                lambda,
                Dual64::from(delta),
                rho[0] + rho[1] * 4.0,
            )
        };
        let h = 1e-6;
        let fd = (g(Dual64::from(7.0 + h)).re - g(Dual64::from(7.0 - h)).re) / (2.0 * h);
        assert_relative_eq!(
            g(Dual64::from(7.0).derivative()).eps,
            fd,
            max_relative = 1e-8
        );
    }
}
