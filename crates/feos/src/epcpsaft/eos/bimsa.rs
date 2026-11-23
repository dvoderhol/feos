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
//!   X'_i = (z_i - η(Γ) d_i²) / (1 + Γ d_i), d_ij = (d_i + d_j) / 2; K_ij association constant ([PairingConstant]:
//!   Bjerrum, Ebeling or contact shell), f_ij logarithm of an adjustable factor (`ion_pairing`), g_ij contact value
//!   of the hard-sphere mixture of all species.
//!
//! ∂F/∂X_i = 0 is the mass action law X_i = 1 / (1 + Σ_j ρ_j X_j Δ_ij); at that solution the pairing part equals
//! Σ_i ρ_i (ln X_i - X_i / 2 + 1/2). ∂F/∂Γ = 0 is the BiMSA closure; for a symmetric salt (equal size and charge
//! magnitude, η = 0) it is 4Γ²(1 + Γσ)² = κ² (α + Γσ) / (1 + Γσ) of Blum and Bernard with the fraction α of non-paired
//! ions. The exponential
//! factor of Δ is the electrostatic contact value of the pair normalized to 1 at infinite dilution. For ions of
//! different size, η is that of the MSA functional (the Bernard-Blum closure uses a modified η).
//!
//! Alternatively ([BiMsaClosure::BernardBlum]), the equations of Bernard and Blum (J. Chem. Phys. 104 (1996) 4746,
//! section IV, eqs. 109, 112-114, 119, 120), in which the pairs also enter η, Ω, the closure for Γ and the Helmholtz
//! energy ([bimsa_bernard_blum]); the pair term of eq. 114 with the prefactor π of the general closure eq. 65.
//! Identical to the variational form for ions of equal size and charge magnitude. Both forms return NaN if the
//! iteration does not converge.
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

/// Closure of the BiMSA.
#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub enum BiMsaClosure {
    /// Stationary point of the variational functional (η of the MSA without pairs).
    #[default]
    Variational,
    /// Equations of Bernard and Blum (1996) with the pairs in η, Ω, Γ and the Helmholtz energy.
    BernardBlum,
}

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
    /// closure
    pub closure: BiMsaClosure,
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

/// Residuals of the stationarity conditions: [∂F/∂Γ, X_i (1 + Σ_j ρ_j X_j Δ_ij(Γ)) - 1].
#[expect(clippy::too_many_arguments)]
fn residuals<T: DualNum<Primitive = f64> + Copy>(
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
pub fn bimsa<D: DualNum<Primitive = f64> + Copy>(input: &BiMsaInput<D>) -> (D, D, DVector<D>) {
    if input.closure == BiMsaClosure::BernardBlum {
        return bimsa_bernard_blum(input);
    }
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
    let Some(inv) = jac.try_inverse() else {
        let nan = D::from(f64::NAN);
        return (nan, nan, DVector::from_element(n, nan));
    };
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

/// BiMSA with the closure of Bernard and Blum ([BiMsaClosure::BernardBlum]). Returns (a / (kT V), Γ, X).
/// Newton iteration of the real parts (Jacobian by central differences, damped so that Γ > 0 and 0 < X <= 1),
/// started from the variational solution, then Newton steps in the number type D with the real Jacobian.
pub fn bimsa_bernard_blum<D: DualNum<Primitive = f64> + Copy>(
    input: &BiMsaInput<D>,
) -> (D, D, DVector<D>) {
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
    let g_r = input.g.map(|x| x.re());
    let f_r = input.f.map(|x| x.re());
    let start = BiMsaInput {
        rho: &rho_r,
        z,
        d: &d_r,
        delta: delta_r,
        lb: lb_r,
        g: &g_r,
        f: &f_r,
        constant: input.constant,
        closure: BiMsaClosure::Variational,
    };
    let (_, gamma0, x0) = bimsa(&start);
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
                closure: BiMsaClosure::Variational,
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
                closure: BiMsaClosure::Variational,
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

    /// Bernard-Blum closure: equal to the variational form for ions of equal size and charge magnitude; the pair
    /// densities enter η for unequal sizes (different result); exact density derivatives (dual numbers against
    /// central differences).
    #[test]
    fn bernard_blum_closure() {
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
            let mut bb = c.input();
            bb.closure = BiMsaClosure::BernardBlum;
            let (a_v, g_v, x_v) = bimsa(&c.input());
            let (a_b, g_b, x_b) = bimsa(&bb);
            assert_relative_eq!(a_v, a_b, max_relative = 1e-11);
            assert_relative_eq!(g_v, g_b, max_relative = 1e-11);
            assert_relative_eq!(x_v[0], x_b[0], max_relative = 1e-11);
            let c = base(vec![2.2, 4.8], rho);
            let mut bb = c.input();
            bb.closure = BiMsaClosure::BernardBlum;
            assert!((bimsa(&c.input()).0 / bimsa(&bb).0 - 1.0).abs() > 1e-8);
        }
        let c = mixture();
        for k in 0..4 {
            let rho_d: Vec<Dual64> = (0..4)
                .map(|i| Dual64::new(c.rho[i], if i == k { 1.0 } else { 0.0 }))
                .collect();
            let d_d: Vec<Dual64> = c.d.iter().map(|&x| Dual64::from(x)).collect();
            let delta = -rho_d
                .iter()
                .zip(&c.d)
                .fold(Dual64::from(0.0), |acc, (&r, &d)| acc + r * d.powi(3))
                * FRAC_PI_6
                + 1.0;
            let (g, f) = (c.g.map(Dual64::from), c.f.map(Dual64::from));
            let input = BiMsaInput {
                rho: &rho_d,
                z: &c.z,
                d: &d_d,
                delta,
                lb: Dual64::from(c.lb),
                g: &g,
                f: &f,
                constant: PairingConstant::Bjerrum,
                closure: BiMsaClosure::BernardBlum,
            };
            let a = bimsa(&input).0;
            let h = 1e-7 * c.rho[k];
            let fd: Vec<f64> = [h, -h]
                .iter()
                .map(|&dh| {
                    let mut r = c.rho.clone();
                    r[k] += dh;
                    let cc = Case {
                        rho: r,
                        ..mixture()
                    };
                    let mut inp = cc.input();
                    inp.closure = BiMsaClosure::BernardBlum;
                    bimsa(&inp).0
                })
                .collect();
            assert_relative_eq!(a.eps, (fd[0] - fd[1]) / (2.0 * h), max_relative = 1e-6);
        }
    }
}
