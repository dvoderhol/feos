//! Binding mean spherical approximation (BiMSA) with the closure of Bernard and Blum (J. Chem. Phys. 104 (1996) 4746,
//! section IV, eqs. 109, 112-114, 119, 120; Blum and Bernard, J. Stat. Phys. 79 (1995) 569; Simonin, Bernard and
//! Blum, J. Phys. Chem. B 102 (1998) 4411, eqs. 1-11).
//!
//! The unknowns are the screening parameter Γ, the parameter η and the fractions X_i of non-paired ions:
//! - Γ² = π λ_B (Σ_k ρ_k X'_k² + Σ_P ρ_P/σ_P (σ_i/u_i + σ_j/u_j) X'_i X'_j),
//! - η = π/(2ΔΩ) (Σ_k ρ_k z_k σ_k/u_k + Σ_P ρ_P (z_i σ_j² + z_j σ_i²) / (2 σ_P u_i u_j)),
//!   Ω = 1 + π/(2Δ) (Σ_k ρ_k σ_k³/u_k + Σ_P ρ_P σ_i² σ_j² / (σ_P u_i u_j)),
//! - X_i (1 + Σ_j ρ_j X_j Δ_ij) = 1 (mass action law),
//!
//! with u_k = 1 + Γσ_k, X'_k = (z_k - η σ_k²)/u_k, the distance of closest approach σ_P = (σ_i + σ_j)/2 and the pair
//! densities ρ_P = ρ_i ρ_j X_i X_j Δ_ij of all cation-anion pairs P = (i, j); for one pair these are eqs. 112-114 of
//! Bernard and Blum with the prefactor π of eq. 65 (eqs. 4-6 of Simonin et al. 1998, who denote σ_i + σ_j by σ_P);
//! for several pairs the pair terms are summed. The association strength
//! is Δ_ij = exp(f_ij) K_ij g_ij exp[-λ_B (X'_i X'_j - z_i z_j) / σ_P] with the association constant K_ij
//! ([PairingConstant]: Bjerrum, Ebeling or contact shell), the logarithm f_ij of an adjustable factor (`ion_pairing`)
//! and the contact value g_ij of the hard-sphere mixture of all species.
//!
//! The Helmholtz energy density is -λ_B Σ_k ρ_k z_k (Γ z_k + η σ_k)/u_k + Γ³/(3π) + λ_B η Σ_P ρ_P/(2σ_P) (σ_i² X'_j
//! u_j + σ_j² X'_i u_i)/(u_i u_j) + Σ_k ρ_k (ln X_k - X_k/2 + 1/2) (eqs. 119, 120).
//!
//! The iteration starts from the stationary point of the functional
//! A_MSA(Γ) + Σ_i ρ_i (ln X_i - X_i + 1) - 1/2 Σ_ij ρ_i ρ_j X_i X_j Δ_ij(Γ)
//! with η of the MSA without pairs ([msa_functional]), which coincides with the solution for ions of equal size and
//! charge magnitude. NaN is returned if the iteration does not converge.
use super::ion_pairing::{PairingConstant, non_paired_fractions};
use super::msa::{eta, msa_functional, screening_parameter};
use nalgebra::{DMatrix, DVector};
use num_dual::{Dual, DualNum};
use std::f64::consts::{FRAC_PI_2, PI};

const MAX_ITER: usize = 200;
const TOL: f64 = 1e-14;
/// Relative size of the last Newton step of the Bernard-Blum iteration at convergence.
const BB_TOL: f64 = 1e-12;
/// Newton steps of the joint system in the number type D after the real parts converged.
const EXTRA_ITERATIONS: usize = 5;

/// Inputs of the BiMSA term that do not depend on Γ and X.
pub struct BiMsaInput<'a, D> {
    pub rho: &'a [D],
    pub z: &'a [f64],
    pub d: &'a [D],
    /// packing term Δ of the MSA
    pub delta: D,
    pub lb: D,
    /// contact values of the hard-sphere mixture at (d_i + d_j)/2
    pub g: &'a DMatrix<D>,
    /// logarithms of the pairing factors
    pub f: &'a DMatrix<D>,
    /// association constant
    pub constant: PairingConstant,
    /// binding distance factor θ of the contact-shell constant ([PairingConstant::constant])
    pub shell_factor: D,
}

impl<D: DualNum<Primitive = f64> + Copy> BiMsaInput<'_, D> {
    /// e^f_ij K_ij g_ij (Γ-independent part of Δ_ij)
    fn prefactor(&self) -> DMatrix<D> {
        let n = self.rho.len();
        DMatrix::from_fn(n, n, |i, j| {
            if self.z[i] * self.z[j] >= 0.0 {
                return D::zero();
            }
            let k = self.constant.constant(
                self.lb * (self.z[i] * self.z[j]).abs(),
                (self.d[i] + self.d[j]) * 0.5,
                self.shell_factor,
            );
            self.f[(i, j)].exp() * k * self.g[(i, j)]
        })
    }
}

/// Δ_ij(Γ) for the prefactors `pre` (generic in the number type of Γ).
fn strength<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> DMatrix<T> {
    let n = rho.len();
    let eta = eta(gamma, rho, z, d, delta);
    let xp: Vec<T> = (0..n)
        .map(|i| (-eta * d[i] * d[i] + z[i]) / (gamma * d[i] + 1.0))
        .collect();
    DMatrix::from_fn(n, n, |i, j| {
        if pre[(i, j)].re() == 0.0 {
            return T::zero();
        }
        let dij = (d[i] + d[j]) * 0.5;
        pre[(i, j)] * (-lb * (xp[i] * xp[j] - z[i] * z[j]) / dij).exp()
    })
}

/// F(Γ, X) / (kT V) (generic in the number type).
#[expect(clippy::too_many_arguments)]
fn functional<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> T {
    let n = rho.len();
    let dm = strength(gamma, rho, z, d, delta, lb, pre);
    let mut pair = T::zero();
    for i in 0..n {
        pair += rho[i] * (x[i].ln() - x[i] + 1.0);
        for j in 0..n {
            pair -= rho[i] * rho[j] * x[i] * x[j] * dm[(i, j)] * 0.5;
        }
    }
    msa_functional(gamma, rho, z, d, lb, delta) + pair
}

/// ∂F/∂Γ at fixed X (in the number type T).
#[expect(clippy::too_many_arguments)]
fn dfunctional_dgamma<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> T {
    let l = |v: T| Dual::<T>::from_re(v);
    let rho_l: Vec<_> = rho.iter().map(|&v| l(v)).collect();
    let d_l: Vec<_> = d.iter().map(|&v| l(v)).collect();
    let x_l: Vec<_> = x.iter().map(|&v| l(v)).collect();
    functional(
        Dual::new(gamma, T::one()),
        &x_l,
        &rho_l,
        z,
        &d_l,
        l(delta),
        l(lb),
        &pre.map(l),
    )
    .eps
}

/// Starting values (Γ, X) of the Bernard-Blum iteration: stationary point of the functional F(Γ, X) with ∂F/∂X_i = 0
/// (mass action law) and ∂F/∂Γ = 0 (bisection in Γ; the pairing term lowers Γ below the MSA value).
fn starting_values(
    rho: &[f64],
    z: &[f64],
    d: &[f64],
    delta: f64,
    lb: f64,
    pre: &DMatrix<f64>,
    sum_rho_z2: f64,
) -> (f64, DVector<f64>) {
    let gamma_msa = screening_parameter(rho, z, d, lb, delta, sum_rho_z2);
    let solve_x = |gamma: f64| {
        let dm = strength(gamma, rho, z, d, delta, lb, pre);
        non_paired_fractions(&DVector::from_column_slice(rho), &dm)
    };
    let h = |gamma: f64| {
        let x = solve_x(gamma);
        (
            dfunctional_dgamma(gamma, x.as_slice(), rho, z, d, delta, lb, pre),
            x,
        )
    };
    // bracket: ∂F/∂Γ < 0 for Γ -> 0, > 0 for large Γ
    let (mut lo, mut hi) = (gamma_msa * 1e-6, gamma_msa);
    while h(hi).0 < 0.0 {
        hi *= 2.0;
    }
    let mut gamma = gamma_msa;
    for _ in 0..MAX_ITER {
        gamma = 0.5 * (lo + hi);
        if h(gamma).0 < 0.0 {
            lo = gamma;
        } else {
            hi = gamma;
        }
        if hi - lo <= TOL * gamma {
            break;
        }
    }
    (gamma, h(gamma).1)
}

/// Residuals of the Bernard-Blum equations for the unknowns v = [Γ, η, X_0, ..., X_(n-1)]: [Γ² - π λ_B (Σ_k ρ_k X'_k² +
/// Σ_P ρ_P/σ_P (σ_i/u_i + σ_j/u_j) X'_i X'_j), η - π/(2ΔΩ) (Σ_k ρ_k z_k σ_k/u_k + Σ_P ρ_P (z_i σ_j² + z_j σ_i²) /
/// (2 σ_P u_i u_j)), X_i (1 + Σ_j ρ_j X_j Δ_ij) - 1] with Ω = 1 + π/(2Δ) (Σ_k ρ_k σ_k³/u_k + Σ_P ρ_P σ_i² σ_j² /
/// (σ_P u_i u_j)), u_k = 1 + Γσ_k, X'_k = (z_k - η σ_k²)/u_k and the pair densities ρ_P = ρ_i ρ_j X_i X_j Δ_ij of all
/// cation-anion pairs P = (i, j) (Bernard and Blum 1996, eqs. 112-114 for one pair; several pairs: sum of the pair
/// terms).
fn bernard_blum_residuals<T: DualNum<Primitive = f64> + Copy>(
    v: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> (DVector<T>, T) {
    let n = rho.len();
    let (gamma, eta) = (v[0], v[1]);
    let x = &v[2..];
    let u: Vec<T> = d.iter().map(|&di| gamma * di + 1.0).collect();
    let xp: Vec<T> = (0..n).map(|k| (-eta * d[k] * d[k] + z[k]) / u[k]).collect();
    let dm = DMatrix::from_fn(n, n, |i, j| {
        if pre[(i, j)].re() == 0.0 {
            return T::zero();
        }
        pre[(i, j)] * (-lb * (xp[i] * xp[j] - z[i] * z[j]) / ((d[i] + d[j]) * 0.5)).exp()
    });
    let (mut om, mut et, mut ga) = (T::zero(), T::zero(), T::zero());
    for k in 0..n {
        om += rho[k] * d[k].powi(3) / u[k];
        et += rho[k] * d[k] * z[k] / u[k];
        ga += rho[k] * xp[k] * xp[k];
    }
    let mut a_pair = T::zero();
    let mut r = DVector::zeros(n + 2);
    for i in 0..n {
        let mut s = T::zero();
        for j in 0..n {
            if dm[(i, j)].re() == 0.0 {
                continue;
            }
            s += rho[j] * x[j] * dm[(i, j)];
            if j > i {
                let rp = rho[i] * rho[j] * x[i] * x[j] * dm[(i, j)];
                let dij = (d[i] + d[j]) * 0.5;
                let uu = u[i] * u[j];
                om += rp * d[i] * d[i] * d[j] * d[j] / (dij * uu);
                et += rp * (d[j] * d[j] * z[i] + d[i] * d[i] * z[j]) / (dij * uu * 2.0);
                ga += rp / dij * (d[i] / u[i] + d[j] / u[j]) * xp[i] * xp[j];
                a_pair += rp / (dij * 2.0)
                    * (d[i] * d[i] * xp[j] * u[j] + d[j] * d[j] * xp[i] * u[i])
                    / uu;
            }
        }
        r[i + 2] = x[i] * (s + 1.0) - 1.0;
    }
    let omega = om * FRAC_PI_2 / delta + 1.0;
    r[0] = gamma * gamma - lb * ga * PI;
    r[1] = eta - et * FRAC_PI_2 / (delta * omega);
    (r, a_pair)
}

/// Helmholtz energy density of the Bernard-Blum BiMSA (eqs. 119, 120) at the solution v of
/// [bernard_blum_residuals]: -λ_B Σ_k ρ_k z_k (Γ z_k + η σ_k)/u_k + Γ³/(3π) + λ_B η Σ_P ρ_P/(2σ_P) (σ_i² X'_j u_j +
/// σ_j² X'_i u_i)/(u_i u_j) + Σ_k ρ_k (ln X_k - X_k/2 + 1/2).
fn bernard_blum_helmholtz<T: DualNum<Primitive = f64> + Copy>(
    v: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    lb: T,
    a_pair: T,
) -> T {
    let (gamma, eta) = (v[0], v[1]);
    let mut a = gamma.powi(3) / (3.0 * PI) + lb * eta * a_pair;
    for k in 0..rho.len() {
        let x = v[k + 2];
        a += -lb * rho[k] * z[k] * (gamma * z[k] + eta * d[k]) / (gamma * d[k] + 1.0)
            + rho[k] * (x.ln() - x * 0.5 + 0.5);
    }
    a
}

/// Reduced Helmholtz energy density of the BiMSA term. Returns (a / (kT V), Γ, fractions of non-paired ions X).
/// Newton iteration of the real parts (Jacobian by central differences, damped so that Γ > 0 and 0 < X <= 1),
/// started from [starting_values], then Newton steps in the number type D with the real Jacobian.
pub fn bimsa<D: DualNum<Primitive = f64> + Copy>(input: &BiMsaInput<D>) -> (D, D, DVector<D>) {
    let BiMsaInput {
        rho,
        z,
        d,
        delta,
        lb,
        ..
    } = *input;
    let n = rho.len();
    let sum_rho_z2 = rho
        .iter()
        .zip(z)
        .fold(D::zero(), |acc, (&r, &z)| acc + r * z * z);
    if sum_rho_z2.re() == 0.0 {
        return (D::zero(), D::zero(), DVector::from_element(n, D::one()));
    }
    let pre = input.prefactor();
    let re = |v: &[D]| v.iter().map(|x| x.re()).collect::<Vec<f64>>();
    let (rho_r, d_r) = (re(rho), re(d));
    let (delta_r, lb_r) = (delta.re(), lb.re());
    let pre_r = pre.map(|x| x.re());
    let (gamma0, x0) = starting_values(&rho_r, z, &d_r, delta_r, lb_r, &pre_r, sum_rho_z2.re());
    let eta0 = eta(gamma0, &rho_r, z, &d_r, delta_r);
    let mut v: Vec<f64> = [gamma0, eta0]
        .into_iter()
        .chain(x0.iter().copied())
        .collect();
    let m = n + 2;
    let res = |v: &[f64]| bernard_blum_residuals(v, &rho_r, z, &d_r, delta_r, lb_r, &pre_r).0;
    let scale: Vec<f64> = (0..m)
        .map(|k| {
            if k == 1 {
                gamma0 / d_r.iter().fold(0.0, |a: f64, &b| a.max(b))
            } else {
                v[k].abs().max(gamma0.min(1.0))
            }
        })
        .collect();
    let jacobian = |v: &[f64]| {
        DMatrix::from_fn(m, m, |i, k| {
            let h = 1e-7 * (v[k].abs() + scale[k]);
            let mut vp = v.to_vec();
            let mut vm = v.to_vec();
            vp[k] += h;
            vm[k] -= h;
            (res(&vp)[i] - res(&vm)[i]) / (2.0 * h)
        })
    };
    let norm = |r: &DVector<f64>| {
        r.iter()
            .zip(&scale)
            .map(|(x, s)| (x / s).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let r = res(&v);
        let Some(inv) = jacobian(&v).try_inverse() else {
            break;
        };
        let step = -(&inv * &r);
        // converged: the full Newton step is below BB_TOL relative to the variable plus its typical size
        if step
            .iter()
            .zip(&v)
            .zip(&scale)
            .all(|((s, x), sc)| s.abs() <= BB_TOL * (x.abs() + sc))
        {
            v.iter_mut().zip(step.iter()).for_each(|(x, s)| *x += s);
            converged = true;
            break;
        }
        let mut t = 1.0;
        loop {
            let trial: Vec<f64> = v.iter().zip(step.iter()).map(|(a, b)| a + t * b).collect();
            let valid = trial[0] > 0.0 && trial[2..].iter().all(|&x| x > 0.0 && x <= 1.0);
            if (valid && norm(&res(&trial)) <= norm(&r)) || t < 1e-6 {
                if valid {
                    v = trial;
                }
                break;
            }
            t *= 0.5;
        }
    }
    let (true, Some(inv)) = (converged, jacobian(&v).try_inverse()) else {
        let nan = D::from(f64::NAN);
        return (nan, nan, DVector::from_element(n, nan));
    };
    let mut w: Vec<D> = v.iter().map(|&x| D::from(x)).collect();
    for _ in 0..EXTRA_ITERATIONS {
        let r = bernard_blum_residuals(&w, rho, z, d, delta, lb, &pre).0;
        let step: Vec<D> = (0..m)
            .map(|i| (0..m).fold(D::zero(), |acc, k| acc + r[k] * inv[(i, k)]))
            .collect();
        for i in 0..m {
            w[i] -= step[i];
        }
    }
    let (_, a_pair) = bernard_blum_residuals(&w, rho, z, d, delta, lb, &pre);
    let a = bernard_blum_helmholtz(&w, rho, z, d, lb, a_pair);
    (a, w[0], DVector::from_column_slice(&w[2..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcpsaft::eos::ion_pairing::bjerrum_constant;
    use approx::assert_relative_eq;
    use num_dual::Dual64;
    use std::f64::consts::{FRAC_PI_6, PI};

    struct Case {
        rho: Vec<f64>,
        z: Vec<f64>,
        d: Vec<f64>,
        g: DMatrix<f64>,
        f: DMatrix<f64>,
        lb: f64,
    }

    impl Case {
        fn delta(&self) -> f64 {
            1.0 - FRAC_PI_6
                * self
                    .rho
                    .iter()
                    .zip(&self.d)
                    .map(|(r, d)| r * d.powi(3))
                    .sum::<f64>()
        }
        fn input(&self) -> BiMsaInput<'_, f64> {
            BiMsaInput {
                rho: &self.rho,
                z: &self.z,
                d: &self.d,
                delta: self.delta(),
                lb: self.lb,
                g: &self.g,
                f: &self.f,
                constant: PairingConstant::Bjerrum,
                shell_factor: 1.0,
            }
        }
    }

    fn mixture() -> Case {
        Case {
            rho: vec![0.003, 0.003, 0.001, 0.001],
            z: vec![2.0, -2.0, 1.0, -1.0],
            d: vec![2.8, 3.4, 2.6, 2.9],
            g: DMatrix::from_element(4, 4, 1.2),
            f: DMatrix::from_element(4, 4, 0.0),
            lb: 7.1,
        }
    }

    /// Without pairing the term equals the MSA (functional at the MSA screening parameter).
    #[test]
    fn no_pairing_is_msa() {
        let mut c = mixture();
        c.f = DMatrix::from_element(4, 4, -200.0);
        let (a, gamma, x) = bimsa(&c.input());
        let s: f64 = c.rho.iter().zip(&c.z).map(|(r, z)| r * z * z).sum();
        let g_msa = screening_parameter(&c.rho, &c.z, &c.d, c.lb, c.delta(), s);
        assert_relative_eq!(gamma, g_msa, max_relative = 1e-10);
        assert_relative_eq!(
            a,
            msa_functional(g_msa, &c.rho, &c.z, &c.d, c.lb, c.delta()),
            max_relative = 1e-12
        );
        assert!(x.iter().all(|&x| (x - 1.0).abs() < 1e-12));
    }

    /// Restricted primitive model (one symmetric salt, equal sizes): Γ satisfies the BiMSA equation of Blum and
    /// Bernard 4Γ²(1 + Γσ)² = κ² (α + Γσ) / (1 + Γσ), α = fraction of non-paired ions.
    #[test]
    fn restricted_primitive_model() {
        let sigma = 4.0;
        let c = Case {
            rho: vec![0.002, 0.002],
            z: vec![2.0, -2.0],
            d: vec![sigma, sigma],
            g: DMatrix::from_element(2, 2, 1.0),
            f: DMatrix::from_element(2, 2, 2.0),
            lb: 7.14,
        };
        let (_, gamma, x) = bimsa(&c.input());
        let alpha = x[0];
        assert!(
            alpha < 0.9,
            "pairing too weak for the test: alpha = {alpha}"
        );
        let kappa2 = 4.0 * PI * c.lb * c.rho.iter().zip(&c.z).map(|(r, z)| r * z * z).sum::<f64>();
        let u = 1.0 + gamma * sigma;
        assert_relative_eq!(
            4.0 * gamma * gamma * u * u,
            kappa2 * (alpha + gamma * sigma) / u,
            max_relative = 1e-10
        );
    }

    /// At infinite dilution Δ -> e^f K g (electrostatic contact factor -> 1).
    #[test]
    fn infinite_dilution() {
        let mut c = mixture();
        c.rho = c.rho.iter().map(|r| r * 1e-12).collect();
        c.g = DMatrix::from_element(4, 4, 1.0);
        let (_, gamma, _) = bimsa(&c.input());
        let dm = strength(
            gamma,
            &c.rho,
            &c.z,
            &c.d,
            c.delta(),
            c.lb,
            &c.input().prefactor(),
        );
        let k = bjerrum_constant(c.lb * 4.0, (c.d[0] + c.d[1]) * 0.5);
        assert_relative_eq!(dm[(0, 1)], k, max_relative = 1e-4);
    }

    /// Stationarity: the derivative with respect to a density from dual numbers (Γ and X solved inside) equals
    /// central differences of the converged value.
    #[test]
    fn derivative_density() {
        let c = mixture();
        for k in 0..4 {
            let rho_d: Vec<Dual64> = (0..4)
                .map(|i| Dual64::new(c.rho[i], if i == k { 1.0 } else { 0.0 }))
                .collect();
            let d_d: Vec<Dual64> = c.d.iter().map(|&x| Dual64::from(x)).collect();
            let delta_d = -rho_d
                .iter()
                .zip(&d_d)
                .fold(Dual64::from(0.0), |acc, (&r, &d)| acc + r * d.powi(3))
                * FRAC_PI_6
                + 1.0;
            let (g, f) = (c.g.map(Dual64::from), c.f.map(Dual64::from));
            let input = BiMsaInput {
                rho: &rho_d,
                z: &c.z,
                d: &d_d,
                delta: delta_d,
                lb: Dual64::from(c.lb),
                g: &g,
                f: &f,
                constant: PairingConstant::Bjerrum,
                shell_factor: Dual64::from(1.0),
            };
            let (a, _, _) = bimsa(&input);
            let h = 1e-7 * c.rho[k];
            let value = |dr: f64| {
                let mut c2 = mixture();
                c2.rho[k] += dr;
                bimsa(&c2.input()).0
            };
            assert_relative_eq!(
                a.eps,
                (value(h) - value(-h)) / (2.0 * h),
                max_relative = 1e-6
            );
        }
    }

    /// Pairing lowers the screening parameter.
    #[test]
    fn pairing_lowers_gamma() {
        let c = mixture();
        let s: f64 = c.rho.iter().zip(&c.z).map(|(r, z)| r * z * z).sum();
        let g_msa = screening_parameter(&c.rho, &c.z, &c.d, c.lb, c.delta(), s);
        let (_, gamma, _) = bimsa(&c.input());
        assert!(gamma < g_msa);
    }

    /// Ions of equal size and charge magnitude: the solution equals the stationary point of the functional
    /// ([starting_values]) and the Helmholtz energy its value; for unequal sizes the pairs enter η (different
    /// result). Derivative with respect to the binding distance factor of the contact-shell constant against
    /// central differences.
    #[test]
    fn equal_sizes_and_shell_factor() {
        let lb = 7.1;
        let base = |d: Vec<f64>, rho: f64| Case {
            rho: vec![rho, rho],
            z: vec![1.0, -1.0],
            d,
            g: DMatrix::from_element(2, 2, 1.2),
            f: DMatrix::from_element(2, 2, 2.0),
            lb,
        };
        for rho in [1e-4, 3e-3] {
            let c = base(vec![3.5, 3.5], rho);
            let input = c.input();
            let pre = input.prefactor();
            let s: f64 = c.rho.iter().zip(&c.z).map(|(r, z)| r * z * z).sum();
            let (g0, x0) = starting_values(&c.rho, &c.z, &c.d, c.delta(), lb, &pre, s);
            let f0 = functional(g0, x0.as_slice(), &c.rho, &c.z, &c.d, c.delta(), lb, &pre);
            let (a, gamma, x) = bimsa(&input);
            assert_relative_eq!(a, f0, max_relative = 1e-11);
            assert_relative_eq!(gamma, g0, max_relative = 1e-11);
            assert_relative_eq!(x[0], x0[0], max_relative = 1e-11);
            let c = base(vec![2.2, 4.8], rho);
            let input = c.input();
            let pre = input.prefactor();
            let (g0, x0) = starting_values(&c.rho, &c.z, &c.d, c.delta(), lb, &pre, s);
            let f0 = functional(g0, x0.as_slice(), &c.rho, &c.z, &c.d, c.delta(), lb, &pre);
            assert!((bimsa(&input).0 / f0 - 1.0).abs() > 1e-8);
        }
        let c = mixture();
        let value = |theta: Dual64| {
            let rho: Vec<Dual64> = c.rho.iter().map(|&x| Dual64::from(x)).collect();
            let d: Vec<Dual64> = c.d.iter().map(|&x| Dual64::from(x)).collect();
            let (g, f) = (c.g.map(Dual64::from), c.f.map(Dual64::from));
            let input = BiMsaInput {
                rho: &rho,
                z: &c.z,
                d: &d,
                delta: Dual64::from(c.delta()),
                lb: Dual64::from(c.lb),
                g: &g,
                f: &f,
                constant: PairingConstant::Shell(1.034),
                shell_factor: theta,
            };
            bimsa(&input).0
        };
        let theta = 1.034;
        let h = 1e-6;
        let fd =
            (value(Dual64::from(theta + h)).re - value(Dual64::from(theta - h)).re) / (2.0 * h);
        assert_relative_eq!(value(Dual64::new(theta, 1.0)).eps, fd, max_relative = 1e-6);
    }
}
