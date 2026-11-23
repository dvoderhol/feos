//! Building blocks of ion pairing between cations and anions: the Bjerrum association constant and the
//! Wertheim (TPT1) association of one pairing site per ion.
//!
//! Every ion has one pairing site that can bind to the site of an oppositely charged ion; X_i is the
//! fraction of ions i that are not paired. With the association strengths Δ_ij (Å³, number-density based,
//! Δ_ij = 0 for ions of the same charge sign),
//!
//! X_i = 1 / (1 + Σ_j ρ_j X_j Δ_ij),
//! a / (kT V) = Σ_i ρ_i (ln X_i - X_i / 2 + 1/2),
//!
//! and the density of pairs of ions i and j is ρ_i ρ_j X_i X_j Δ_ij.
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::PI;

/// Euler-Mascheroni constant
#[expect(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// Maximum number of terms of the series of the exponential integral.
const EI_MAX_TERMS: usize = 400;
const MAX_ITER: usize = 200;
const TOL: f64 = 1e-14;
/// Newton iterations in the number type D after the real parts converged.
const EXTRA_ITERATIONS: usize = 4;

/// Exponential integral Ei(t) for t > 0 from the series Ei(t) = γ + ln t + Σ_{k≥1} t^k / (k k!).
///
/// The series converges for all t; the terms are evaluated in the number type D, so that all derivatives
/// are those of the series (truncated where the real part of a term is below 1e-17 of the sum).
pub fn exponential_integral<D: DualNum<Primitive = f64> + Copy>(t: D) -> D {
    let mut term = D::one(); // t^k / k!
    let mut sum = D::zero();
    for k in 1..=EI_MAX_TERMS {
        term = term * t / k as f64;
        let add = term / k as f64;
        sum += add;
        if add.re().abs() < 1e-17 * sum.re().abs() {
            break;
        }
    }
    sum + t.ln() + EULER_GAMMA
}

/// G(t) = ∫ e^t / t⁴ dt = Ei(t) / 6 - e^t (t² + t + 2) / (6 t³).
fn bjerrum_primitive<D: DualNum<Primitive = f64> + Copy>(t: D) -> D {
    exponential_integral(t) / 6.0 - t.exp() * (t * t + t + 2.0) / (t.powi(3) * 6.0)
}

/// Bjerrum association constant of an ion pair (Å³, per pair, number-density based):
///
/// K = 4π ∫_σ^{l/2} r² exp(l / r) dr = 4π l³ [G(l/σ) - G(2)],   K = 0 for l / σ <= 2,
///
/// with l = |z_i z_j| λ_B (Å) and the distance of closest approach σ (Å). In molar units,
/// K / (L/mol) = K / Å³ × N_A × 1e-27.
pub fn bjerrum_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D) -> D {
    let b = l / sigma;
    if b.re() <= 2.0 {
        return D::zero();
    }
    l.powi(3) * (bjerrum_primitive(b) - bjerrum_primitive(D::from(2.0))) * (4.0 * PI)
}

/// Association constant of contact pairs (Å³, per pair, number-density based): the Bjerrum integral over the shell
/// from the distance of closest approach σ to the binding distance θ σ (Naseri Boroujeni et al., J. Chem. Phys. 161
/// (2024) 057103: θ = 1.034),
///
/// K = 4π ∫_σ^{θσ} r² exp(l / r) dr = 4π l³ [G(l/σ) - G(l/(θσ))],
///
/// with l = |z_i z_j| λ_B (Å); positive for all l > 0.
pub fn shell_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D, theta: f64) -> D {
    l.powi(3) * (bjerrum_primitive(l / sigma) - bjerrum_primitive(l / (sigma * theta))) * (4.0 * PI)
}

/// Ebeling association constant of an ion pair (Å³, per pair, number-density based; Ebeling, Z. Phys. Chem.
/// (Leipzig) 238 (1968) 400):
///
/// K = 8π σ³ Σ_{m≥2} b^{2m} / ((2m)! (2m - 3)),   b = l / σ,
///
/// with l = |z_i z_j| λ_B (Å) and the distance of closest approach σ (Å). K is analytic in b (no cutoff), of order
/// b⁴ for small b and equal to the Bjerrum constant to leading order for large b.
pub fn ebeling_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D) -> D {
    let b = l / sigma;
    let b2 = b * b;
    // term_m = b^{2m} / (2m)!
    let mut term = b2 * b2 / 24.0;
    let mut sum = term;
    for m in 3..=EI_MAX_TERMS {
        let k = (2 * m) as f64;
        term = term * b2 / (k * (k - 1.0));
        let add = term / (k - 3.0);
        sum += add;
        if add.re().abs() < 1e-17 * sum.re().abs() {
            break;
        }
    }
    sigma.powi(3) * sum * (8.0 * PI)
}

/// Association constant of an ion pair used by the ion-pairing variants.
#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub enum PairingConstant {
    /// Bjerrum constant ([bjerrum_constant]; zero for l / σ <= 2).
    #[default]
    Bjerrum,
    /// Ebeling constant ([ebeling_constant]; analytic in l / σ).
    Ebeling,
    /// Contact-pair constant over the shell σ..θσ ([shell_constant]) with the given θ.
    Shell(f64),
}

impl PairingConstant {
    /// Association constant (Å³) for l = |z_i z_j| λ_B and the distance of closest approach σ.
    pub fn constant<D: DualNum<Primitive = f64> + Copy>(&self, l: D, sigma: D) -> D {
        match self {
            Self::Bjerrum => bjerrum_constant(l, sigma),
            Self::Ebeling => ebeling_constant(l, sigma),
            Self::Shell(theta) => shell_constant(l, sigma, *theta),
        }
    }
}

/// Fractions of non-paired ions X_i for the densities `rho` (Å⁻³) and association strengths `delta` (Å³).
///
/// Damped successive substitution in f64, Newton iteration in f64 to convergence (damped so that 0 < X <= 1; the
/// substitution alone converges slowly for strong pairing with unequal densities), then Newton steps in the number
/// type D: the result has the derivatives of the solution of X_i (1 + Σ_j ρ_j X_j Δ_ij) = 1 with respect to
/// everything that D carries. NaN if the iteration does not converge.
pub fn non_paired_fractions<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    delta: &DMatrix<D>,
) -> DVector<D> {
    let n = rho.len();
    let rho_re = rho.map(|r| r.re());
    let delta_re = delta.map(|d| d.re());
    let mut x = DVector::from_element(n, 1.0);
    for _ in 0..MAX_ITER {
        let x_new = DVector::from_fn(n, |i, _| {
            1.0 / (1.0
                + (0..n)
                    .map(|j| rho_re[j] * x[j] * delta_re[(i, j)])
                    .sum::<f64>())
        });
        // damped substitution (stable for strong association)
        let x_next = (&x_new + &x) * 0.5;
        let change = (&x_next - &x).amax();
        x = x_next;
        if change < TOL {
            break;
        }
    }
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let s = DVector::from_fn(n, |i, _| {
            (0..n).fold(0.0, |acc, j| acc + rho_re[j] * x[j] * delta_re[(i, j)])
        });
        let g = DVector::from_fn(n, |i, _| x[i] * (s[i] + 1.0) - 1.0);
        if g.amax() < TOL {
            converged = true;
            break;
        }
        let jac = DMatrix::from_fn(n, n, |i, j| {
            let diag = if i == j { s[i] + 1.0 } else { 0.0 };
            diag + x[i] * rho_re[j] * delta_re[(i, j)]
        });
        let Some(inv) = jac.try_inverse() else { break };
        let step = inv * g;
        let mut t = 1.0;
        while (0..n).any(|i| {
            let xi = x[i] - t * step[i];
            xi <= 0.0 || xi > 1.0
        }) && t > 1e-12
        {
            t *= 0.5;
        }
        x -= step * t;
    }
    if !converged {
        return DVector::from_element(n, D::from(f64::NAN));
    }
    newton(&x.map(D::from), rho, delta, MAX_ITER, true)
}

/// Newton iteration for g_i(X) = X_i (1 + Σ_j ρ_j X_j Δ_ij) - 1 = 0 in the number type D with the Jacobian of the
/// real parts: every step at the converged real solution makes one more order of the derivatives exact; with
/// `converged_start`, EXTRA_ITERATIONS steps.
fn newton<D: DualNum<Primitive = f64> + Copy>(
    x0: &DVector<D>,
    rho: &DVector<D>,
    delta: &DMatrix<D>,
    max_iter: usize,
    converged_start: bool,
) -> DVector<D> {
    let n = rho.len();
    let mut x = x0.clone();
    let iterations = if converged_start {
        EXTRA_ITERATIONS
    } else {
        max_iter
    };
    for _ in 0..iterations {
        let s = DVector::from_fn(n, |i, _| {
            (0..n).fold(D::zero(), |acc, j| acc + rho[j] * x[j] * delta[(i, j)])
        });
        let g = DVector::from_fn(n, |i, _| x[i] * (s[i] + 1.0) - 1.0);
        let jac = DMatrix::from_fn(n, n, |i, j| {
            let diag = if i == j { s[i].re() + 1.0 } else { 0.0 };
            diag + x[i].re() * rho[j].re() * delta[(i, j)].re()
        });
        let Some(inv) = jac.try_inverse() else {
            return DVector::from_element(n, D::from(f64::NAN));
        };
        let step = DVector::from_fn(n, |i, _| {
            (0..n).fold(D::zero(), |acc, j| acc + g[j] * inv[(i, j)])
        });
        x -= step;
    }
    x
}

/// Reduced Helmholtz energy density of the pairing a / (kT V) = Σ_i ρ_i (ln X_i - X_i / 2 + 1/2).
pub fn pairing_helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    x: &DVector<D>,
) -> D {
    rho.iter().zip(x.iter()).fold(D::zero(), |acc, (&r, &x)| {
        acc + r * (x.ln() - x * 0.5 + 0.5)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;
    use nalgebra::dvector;
    use num_dual::{Dual64, first_derivative};

    /// Strong pairing with unequal densities (ρΔ up to 1e8): converged to the closed form of a single salt,
    /// ρ_P = 2 K ρ_+ ρ_- / (1 + K (ρ_+ + ρ_-) + [1 + 2 K (ρ_+ + ρ_-) + K² (ρ_+ - ρ_-)²]^1/2).
    #[test]
    fn strong_pairing_unequal_densities() {
        for (k, ratio) in [
            (1e3, 1.01),
            (1e5, 1.01),
            (1e6, 1.01),
            (1e8, 1.5),
            (1e6, 1.0),
        ] {
            let (rp, rm) = (1e-2 * ratio, 1e-2);
            let rho = dvector![rp, rm];
            let delta = dmatrix![0.0, k; k, 0.0];
            let x = non_paired_fractions(&rho, &delta);
            let s = 1.0 + 2.0 * k * (rp + rm) + (k * (rp - rm)).powi(2);
            let pair = 2.0 * k * rp * rm / (1.0 + k * (rp + rm) + s.sqrt());
            assert_relative_eq!(x[0], 1.0 - pair / rp, max_relative = 1e-8);
            assert_relative_eq!(x[1], 1.0 - pair / rm, max_relative = 1e-8);
        }
    }

    #[test]
    fn exponential_integral_values() {
        // reference values: Abramowitz and Stegun, Table 5.1 / mpmath ei()
        assert_relative_eq!(
            exponential_integral(1.0),
            1.895_117_816_355_936_8,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            exponential_integral(2.0),
            4.954_234_356_001_89,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            exponential_integral(10.0),
            2_492.228_976_241_878,
            max_relative = 1e-13
        );
        assert_relative_eq!(
            exponential_integral(30.0),
            3.689_732_094_072_74e11,
            max_relative = 1e-12
        );
    }

    #[test]
    fn exponential_integral_derivative() {
        // dEi/dt = e^t / t
        for t in [0.5, 2.0, 7.0, 25.0] {
            let (_, d) = first_derivative(exponential_integral, t);
            assert_relative_eq!(d, f64::exp(t) / t, max_relative = 1e-12);
        }
    }

    fn bjerrum_quadrature(l: f64, sigma: f64) -> f64 {
        // composite Simpson rule of 4π ∫_σ^{l/2} r² exp(l/r) dr
        let (a, b) = (sigma, l / 2.0);
        if b <= a {
            return 0.0;
        }
        let n = 200_000;
        let h = (b - a) / n as f64;
        let f = |r: f64| r * r * (l / r).exp();
        let mut s = f(a) + f(b);
        for k in 1..n {
            s += f(a + k as f64 * h) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        4.0 * PI * s * h / 3.0
    }

    #[test]
    fn bjerrum_constant_quadrature() {
        for (l, sigma) in [(7.1, 3.0), (14.3, 3.0), (28.6, 4.0), (64.0, 3.5)] {
            assert_relative_eq!(
                bjerrum_constant(l, sigma),
                bjerrum_quadrature(l, sigma),
                max_relative = 1e-9
            );
        }
        assert_eq!(bjerrum_constant(5.0, 3.0), 0.0);
    }

    #[test]
    fn bjerrum_constant_derivative() {
        // dK/dl at fixed σ against central differences
        let sigma = 3.0;
        for l in [7.0, 15.0, 40.0] {
            let (_, d) = first_derivative(|l: Dual64| bjerrum_constant(l, Dual64::from(sigma)), l);
            let h = 1e-6 * l;
            let fd = (bjerrum_constant(l + h, sigma) - bjerrum_constant(l - h, sigma)) / (2.0 * h);
            assert_relative_eq!(d, fd, max_relative = 1e-6);
        }
    }

    /// Shell constant against the trapezoidal rule with many points.
    #[test]
    fn shell_constant_quadrature() {
        for (l, sigma) in [(7.4, 4.0), (10.19, 3.495), (28.6, 3.5)] {
            let theta = 1.034;
            let n = 200001;
            let h = (theta * sigma - sigma) / (n - 1) as f64;
            let f = |r: f64| r * r * (l / r).exp();
            let mut sum = 0.5 * (f(sigma) + f(theta * sigma));
            for k in 1..n - 1 {
                sum += f(sigma + k as f64 * h);
            }
            let reference = 4.0 * PI * sum * h;
            assert_relative_eq!(
                shell_constant(l, sigma, theta),
                reference,
                max_relative = 1e-9
            );
        }
    }

    /// Ebeling: series against a direct sum, smooth at l/σ = 2, equal to the Bjerrum constant for large l/σ.
    #[test]
    fn ebeling_constant_values() {
        let direct = |b: f64| {
            let mut s = 0.0;
            let mut f = 24.0; // (2m)! for m = 2
            for m in 2..80_i32 {
                if m > 2 {
                    f *= (2 * m) as f64 * (2 * m - 1) as f64;
                }
                s += b.powi(2 * m) / (f * (2 * m - 3) as f64);
            }
            8.0 * PI * s
        };
        for b in [1.0, 2.0, 3.0, 6.0] {
            assert_relative_eq!(ebeling_constant(b, 1.0), direct(b), max_relative = 1e-12);
        }
        // derivative continuous at b = 2 (Bjerrum: jump from 0 to 4π l³ e²/16)
        let d = |b: f64| first_derivative(|x: Dual64| ebeling_constant(x, Dual64::from(1.0)), b).1;
        assert_relative_eq!(d(2.0 - 1e-6), d(2.0 + 1e-6), max_relative = 1e-5);
        // large coupling: Ebeling -> Bjerrum
        assert_relative_eq!(
            ebeling_constant(20.0, 1.0),
            bjerrum_constant(20.0, 1.0),
            max_relative = 2e-4
        );
    }

    #[test]
    fn single_salt_closed_form() {
        // one cation and one anion: ρ_pair = ρ+ ρ- X+ X- Δ equals the closed form of the quadratic mass action
        let (rp, rm, d) = (0.004, 0.002, 150.0);
        let x = non_paired_fractions(&dvector![rp, rm], &dmatrix![0.0, d; d, 0.0]);
        let pair = rp * rm * x[0] * x[1] * d;
        let rt = rp + rm;
        let closed = 2.0 * d * rp * rm
            / (1.0 + d * rt + (1.0 + 2.0 * d * rt + d * d * (rp - rm).powi(2)).sqrt());
        assert_relative_eq!(pair, closed, max_relative = 1e-12);
        // material balance: free + paired = total
        assert_relative_eq!(rp * x[0] + pair, rp, max_relative = 1e-12);
        assert_relative_eq!(rm * x[1] + pair, rm, max_relative = 1e-12);
    }

    #[test]
    fn non_paired_fraction_derivative() {
        // dX/dρ+ from the Newton steps in dual numbers against central differences
        let (rm, d) = (0.003, 400.0);
        let f = |rp: Dual64| {
            non_paired_fractions(
                &dvector![rp, Dual64::from(rm)],
                &dmatrix![Dual64::from(0.0), Dual64::from(d); Dual64::from(d), Dual64::from(0.0)],
            )[0]
        };
        let rp = 0.002;
        let (_, dx) = first_derivative(f, rp);
        let h = 1e-8;
        let g = |rp: f64| non_paired_fractions(&dvector![rp, rm], &dmatrix![0.0, d; d, 0.0])[0];
        assert_relative_eq!(dx, (g(rp + h) - g(rp - h)) / (2.0 * h), max_relative = 1e-6);
    }
}
