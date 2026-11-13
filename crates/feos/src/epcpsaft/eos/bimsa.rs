//! Binding mean spherical approximation (BiMSA; Blum and Bernard 1995, Bernard and Blum 1996, Simonin et al. 1998)
//! in a variational form.
//!
//! The Helmholtz energy is the stationary value of
//!
//! F(Γ, X) / (kT V) = A_MSA(Γ) + Σ_i ρ_i (ln X_i - X_i + 1) - 1/2 Σ_ij ρ_i ρ_j X_i X_j Δ_ij(Γ)
//!
//! with respect to the screening parameter Γ and the fractions X_i of non-paired ions:
//! - A_MSA(Γ): functional of the unrestricted primitive MSA ([msa_functional]) of all ions (paired or not);
//! - Δ_ij(Γ) = exp(f_ij) K_ij g_ij exp[-λ_B (X'_i X'_j - z_i z_j) / d_ij] for a cation i and an anion j (0 otherwise),
//!   X'_i = (z_i - η(Γ) d_i²) / (1 + Γ d_i), d_ij = (d_i + d_j) / 2; K_ij Bjerrum or Ebeling association constant
//!   ([PairingConstant]), f_ij logarithm of an adjustable factor (`ion_pairing`), g_ij contact value of the
//!   hard-sphere mixture of all species.
//!
//! ∂F/∂X_i = 0 is the mass action law X_i = 1 / (1 + Σ_j ρ_j X_j Δ_ij); at that solution the pairing part equals
//! Σ_i ρ_i (ln X_i - X_i / 2 + 1/2). ∂F/∂Γ = 0 is the BiMSA closure; for ions of equal size (η = 0) it is
//! 4Γ²(1 + Γσ)² = κ² (α + Γσ) / (1 + Γσ) of Blum and Bernard with the fraction α of non-paired ions. The exponential
//! factor of Δ is the electrostatic contact value of the pair normalized to 1 at infinite dilution. For ions of
//! different size, η is that of the MSA functional (the Bernard-Blum closure uses a modified η).
use super::ion_pairing::{PairingConstant, non_paired_fractions};
use super::msa::{eta, msa_functional, screening_parameter};
use nalgebra::{DMatrix, DVector};
use num_dual::{Dual, DualNum};

const MAX_ITER: usize = 200;
const TOL: f64 = 1e-14;
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
    /// association constant (Bjerrum or Ebeling)
    pub constant: PairingConstant,
}

impl<D: DualNum<f64> + Copy> BiMsaInput<'_, D> {
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
            );
            self.f[(i, j)].exp() * k * self.g[(i, j)]
        })
    }
}

/// Δ_ij(Γ) for the prefactors `pre` (generic in the number type of Γ).
fn strength<T: DualNum<f64> + Copy>(
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
fn functional<T: DualNum<f64> + Copy>(
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
fn dfunctional_dgamma<T: DualNum<f64> + Copy>(
    gamma: T,
    x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> T {
    let l = |v: T| Dual::<T, f64>::from_re(v);
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

/// Residuals of the stationarity conditions: [∂F/∂Γ, X_i (1 + Σ_j ρ_j X_j Δ_ij(Γ)) - 1].
#[expect(clippy::too_many_arguments)]
fn residuals<T: DualNum<f64> + Copy>(
    gamma: T,
    x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &DMatrix<T>,
) -> DVector<T> {
    let n = rho.len();
    let dm = strength(gamma, rho, z, d, delta, lb, pre);
    let mut r = DVector::zeros(n + 1);
    r[0] = dfunctional_dgamma(gamma, x, rho, z, d, delta, lb, pre);
    for i in 0..n {
        let s = (0..n).fold(T::zero(), |acc, j| acc + rho[j] * x[j] * dm[(i, j)]);
        r[i + 1] = x[i] * (s + 1.0) - 1.0;
    }
    r
}

/// Reduced Helmholtz energy density of the BiMSA term. Returns (a / (kT V), Γ, fractions of non-paired ions X).
pub fn bimsa<D: DualNum<f64> + Copy>(input: &BiMsaInput<D>) -> (D, D, DVector<D>) {
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
    // real parts
    let re = |v: &[D]| v.iter().map(|x| x.re()).collect::<Vec<f64>>();
    let (rho_r, d_r) = (re(rho), re(d));
    let (delta_r, lb_r) = (delta.re(), lb.re());
    let pre_r = pre.map(|x| x.re());
    let gamma_msa = screening_parameter(&rho_r, z, &d_r, lb_r, delta_r, sum_rho_z2.re());
    let solve_x = |gamma: f64| {
        let dm = strength(gamma, &rho_r, z, &d_r, delta_r, lb_r, &pre_r);
        non_paired_fractions(&DVector::from_column_slice(&rho_r), &dm)
    };
    let h = |gamma: f64| {
        let x = solve_x(gamma);
        (
            dfunctional_dgamma(gamma, x.as_slice(), &rho_r, z, &d_r, delta_r, lb_r, &pre_r),
            x,
        )
    };
    // bracket: ∂F/∂Γ < 0 for Γ -> 0, > 0 for large Γ; Γ <= Γ_MSA (the pairing term lowers Γ)
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
    let x_r = h(gamma).1;

    // joint Newton steps in D with the real Jacobian (central differences of the real residuals)
    let m = n + 1;
    let res_r = |v: &[f64]| residuals(v[0], &v[1..], &rho_r, z, &d_r, delta_r, lb_r, &pre_r);
    let mut v0: Vec<f64> = std::iter::once(gamma).chain(x_r.iter().copied()).collect();
    let jac = DMatrix::from_fn(m, m, |i, k| {
        let hstep = 1e-7 * v0[k].abs().max(1e-12);
        let mut vp = v0.clone();
        let mut vm = v0.clone();
        vp[k] += hstep;
        vm[k] -= hstep;
        (res_r(&vp)[i] - res_r(&vm)[i]) / (2.0 * hstep)
    });
    let inv = jac.try_inverse().unwrap_or_else(|| DMatrix::identity(m, m));
    let mut v: Vec<D> = v0.drain(..).map(D::from).collect();
    for _ in 0..EXTRA_ITERATIONS {
        let r = residuals(v[0], &v[1..], rho, z, d, delta, lb, &pre);
        let step: Vec<D> = (0..m)
            .map(|i| (0..m).fold(D::zero(), |acc, k| acc + r[k] * inv[(i, k)]))
            .collect();
        for i in 0..m {
            v[i] -= step[i];
        }
    }
    let (gamma, x) = (v[0], DVector::from_column_slice(&v[1..]));
    (
        functional(gamma, x.as_slice(), rho, z, d, delta, lb, &pre),
        gamma,
        x,
    )
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
}
