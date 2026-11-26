//! Binding mean spherical approximation (BiMSA) with the closure of Bernard and Blum (J. Chem. Phys. 104 (1996) 4746,
//! section IV, eqs. 109, 112-114, 119, 120; Blum and Bernard, J. Stat. Phys. 79 (1995) 569; Simonin, Bernard and
//! Blum, J. Phys. Chem. B 102 (1998) 4411, eqs. 1-11).
//!
//! The unknowns are the screening parameter Γ, the parameter η and the fractions X_i of non-paired ions (the
//! iteration is over Γ and η, with X_i from the mass action law at every Γ, η; Δ_ij and X_i are evaluated as
//! logarithms where their values exceed the range of f64):
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
use super::ion_pairing::{PairingConstant, PairingStrengths, non_paired_fractions, pair_density};
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
    /// binding distance factor θ of the contact-shell constant ([PairingConstant::Shell])
    pub shell_factor: D,
}

impl<D: DualNum<Primitive = f64> + Copy> BiMsaInput<'_, D> {
    /// ln(e^f_ij K_ij g_ij) (Γ-independent part of ln Δ_ij) of every cation-anion pair with K_ij > 0
    fn prefactor(&self) -> PairingStrengths<D> {
        let n = self.rho.len();
        let mut ln_delta = DMatrix::zeros(n, n);
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if self.z[i] * self.z[j] >= 0.0 {
                    continue;
                }
                let Some(ln_k) = self.constant.ln_constant(
                    self.lb * (self.z[i] * self.z[j]).abs(),
                    (self.d[i] + self.d[j]) * 0.5,
                    self.shell_factor,
                ) else {
                    continue;
                };
                let v = self.f[(i, j)] + ln_k + self.g[(i, j)].ln();
                ln_delta[(i, j)] = v;
                ln_delta[(j, i)] = v;
                pairs.push((i, j));
            }
        }
        PairingStrengths { ln_delta, pairs }
    }
}

/// ln Δ_ij(Γ, η) = ln pre_ij - λ_B (X'_i X'_j - z_i z_j) / σ_P for the prefactors `pre` (generic in the number type).
fn strength_at<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    eta: T,
    z: &[f64],
    d: &[T],
    lb: T,
    pre: &PairingStrengths<T>,
) -> PairingStrengths<T> {
    let n = z.len();
    let xp: Vec<T> = (0..n)
        .map(|i| (-eta * d[i] * d[i] + z[i]) / (gamma * d[i] + 1.0))
        .collect();
    let mut s = pre.clone();
    for &(i, j) in &pre.pairs {
        let dij = (d[i] + d[j]) * 0.5;
        let v = pre.ln_delta[(i, j)] - lb * (xp[i] * xp[j] - z[i] * z[j]) / dij;
        s.ln_delta[(i, j)] = v;
        s.ln_delta[(j, i)] = v;
    }
    s
}

/// ln Δ_ij(Γ) with η of the MSA without pairs ([eta]).
fn strength<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &PairingStrengths<T>,
) -> PairingStrengths<T> {
    strength_at(gamma, eta(gamma, rho, z, d, delta), z, d, lb, pre)
}

/// F(Γ, X) / (kT V) for the logarithms `ln_x` of X (generic in the number type).
#[expect(clippy::too_many_arguments)]
fn functional<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    ln_x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &PairingStrengths<T>,
) -> T {
    let s = strength(gamma, rho, z, d, delta, lb, pre);
    let mut pair = T::zero();
    for i in 0..rho.len() {
        pair += rho[i] * (ln_x[i] - ln_x[i].exp() + 1.0);
    }
    for &(i, j) in &s.pairs {
        pair -= pair_density(rho[i], rho[j], ln_x[i] + ln_x[j] + s.ln_delta[(i, j)]);
    }
    msa_functional(gamma, rho, z, d, lb, delta) + pair
}

/// ∂F/∂Γ at fixed X (in the number type T).
#[expect(clippy::too_many_arguments)]
fn dfunctional_dgamma<T: DualNum<Primitive = f64> + Copy>(
    gamma: T,
    ln_x: &[T],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &PairingStrengths<T>,
) -> T {
    let l = |v: T| Dual::<T>::from_re(v);
    let rho_l: Vec<_> = rho.iter().map(|&v| l(v)).collect();
    let d_l: Vec<_> = d.iter().map(|&v| l(v)).collect();
    let ln_x_l: Vec<_> = ln_x.iter().map(|&v| l(v)).collect();
    functional(
        Dual::new(gamma, T::one()),
        &ln_x_l,
        &rho_l,
        z,
        &d_l,
        l(delta),
        l(lb),
        &pre.map(l),
    )
    .eps
}

/// Starting value of Γ of the Bernard-Blum iteration and ln X there: stationary point of the functional F(Γ, X) with
/// ∂F/∂X_i = 0 (mass action law) and ∂F/∂Γ = 0 (bisection in Γ; the pairing term lowers Γ below the MSA value).
fn starting_values(
    rho: &[f64],
    z: &[f64],
    d: &[f64],
    delta: f64,
    lb: f64,
    pre: &PairingStrengths<f64>,
    sum_rho_z2: f64,
) -> (f64, DVector<f64>) {
    let gamma_msa = screening_parameter(rho, z, d, lb, delta, sum_rho_z2);
    let h = |gamma: f64| {
        let s = strength(gamma, rho, z, d, delta, lb, pre);
        let (_, ln_x) = non_paired_fractions(&DVector::from_column_slice(rho), &s);
        (
            dfunctional_dgamma(gamma, ln_x.as_slice(), rho, z, d, delta, lb, pre),
            ln_x,
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

/// Solution of the mass action law and the sums of the Bernard-Blum equations at (Γ, η).
struct BernardBlum<T> {
    /// residuals [Γ² - π λ_B (Σ_k ρ_k X'_k² + Σ_P ρ_P/σ_P (σ_i/u_i + σ_j/u_j) X'_i X'_j),
    /// η - π/(2ΔΩ) (Σ_k ρ_k z_k σ_k/u_k + Σ_P ρ_P (z_i σ_j² + z_j σ_i²) / (2 σ_P u_i u_j))]
    residuals: [T; 2],
    /// Σ_P ρ_P/(2σ_P) (σ_i² X'_j u_j + σ_j² X'_i u_i)/(u_i u_j)
    a_pair: T,
    x: DVector<T>,
    ln_x: DVector<T>,
}

/// Bernard-Blum equations at (Γ, η) = `v` with X_i from the mass action law X_i (1 + Σ_j ρ_j X_j Δ_ij) = 1 at Δ(Γ, η)
/// ([non_paired_fractions]): Ω = 1 + π/(2Δ) (Σ_k ρ_k σ_k³/u_k + Σ_P ρ_P σ_i² σ_j² / (σ_P u_i u_j)), u_k = 1 + Γσ_k,
/// X'_k = (z_k - η σ_k²)/u_k and the pair densities ρ_P = ρ_i ρ_j X_i X_j Δ_ij of all cation-anion pairs P = (i, j)
/// (Bernard and Blum 1996, eqs. 112-114 for one pair; several pairs: sum of the pair terms).
fn bernard_blum<T: DualNum<Primitive = f64> + Copy>(
    v: [T; 2],
    rho: &[T],
    z: &[f64],
    d: &[T],
    delta: T,
    lb: T,
    pre: &PairingStrengths<T>,
) -> BernardBlum<T> {
    let n = rho.len();
    let [gamma, eta] = v;
    let u: Vec<T> = d.iter().map(|&di| gamma * di + 1.0).collect();
    let xp: Vec<T> = (0..n).map(|k| (-eta * d[k] * d[k] + z[k]) / u[k]).collect();
    let s = strength_at(gamma, eta, z, d, lb, pre);
    let (x, ln_x) = non_paired_fractions(&DVector::from_column_slice(rho), &s);
    let (mut om, mut et, mut ga) = (T::zero(), T::zero(), T::zero());
    for k in 0..n {
        om += rho[k] * d[k].powi(3) / u[k];
        et += rho[k] * d[k] * z[k] / u[k];
        ga += rho[k] * xp[k] * xp[k];
    }
    let mut a_pair = T::zero();
    for &(i, j) in &s.pairs {
        let rp = pair_density(rho[i], rho[j], ln_x[i] + ln_x[j] + s.ln_delta[(i, j)]);
        let dij = (d[i] + d[j]) * 0.5;
        let uu = u[i] * u[j];
        om += rp * d[i] * d[i] * d[j] * d[j] / (dij * uu);
        et += rp * (d[j] * d[j] * z[i] + d[i] * d[i] * z[j]) / (dij * uu * 2.0);
        ga += rp / dij * (d[i] / u[i] + d[j] / u[j]) * xp[i] * xp[j];
        a_pair += rp / (dij * 2.0) * (d[i] * d[i] * xp[j] * u[j] + d[j] * d[j] * xp[i] * u[i]) / uu;
    }
    let omega = om * FRAC_PI_2 / delta + 1.0;
    BernardBlum {
        residuals: [
            gamma * gamma - lb * ga * PI,
            eta - et * FRAC_PI_2 / (delta * omega),
        ],
        a_pair,
        x,
        ln_x,
    }
}

/// Helmholtz energy density of the Bernard-Blum BiMSA (eqs. 119, 120) at the solution (Γ, η) = `v`:
/// -λ_B Σ_k ρ_k z_k (Γ z_k + η σ_k)/u_k + Γ³/(3π) + λ_B η Σ_P ρ_P/(2σ_P) (σ_i² X'_j u_j + σ_j² X'_i u_i)/(u_i u_j)
/// + Σ_k ρ_k (ln X_k - X_k/2 + 1/2).
fn bernard_blum_helmholtz<T: DualNum<Primitive = f64> + Copy>(
    v: [T; 2],
    rho: &[T],
    z: &[f64],
    d: &[T],
    lb: T,
    bb: &BernardBlum<T>,
) -> T {
    let [gamma, eta] = v;
    let mut a = gamma.powi(3) / (3.0 * PI) + lb * eta * bb.a_pair;
    for k in 0..rho.len() {
        a += -lb * rho[k] * z[k] * (gamma * z[k] + eta * d[k]) / (gamma * d[k] + 1.0)
            + rho[k] * (bb.ln_x[k] - bb.x[k] * 0.5 + 0.5);
    }
    a
}

/// Reduced Helmholtz energy density of the BiMSA term. Returns (a / (kT V), Γ, fractions of non-paired ions X).
/// Newton iteration of the real parts of (Γ, η) with X from the mass action law at every (Γ, η) (Jacobian by central
/// differences, damped so that Γ > 0), started from [starting_values], then Newton steps in the number type D with the
/// real Jacobian.
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
    let (gamma0, _) = starting_values(&rho_r, z, &d_r, delta_r, lb_r, &pre_r, sum_rho_z2.re());
    let mut v = [gamma0, eta(gamma0, &rho_r, z, &d_r, delta_r)];
    let res = |v: [f64; 2]| {
        let r = bernard_blum(v, &rho_r, z, &d_r, delta_r, lb_r, &pre_r).residuals;
        DVector::from_column_slice(&r)
    };
    let scale = [gamma0, gamma0 / d_r.iter().fold(0.0, |a: f64, &b| a.max(b))];
    let jacobian = |v: [f64; 2]| {
        DMatrix::from_fn(2, 2, |i, k| {
            let h = 1e-7 * (v[k].abs() + scale[k]);
            let (mut vp, mut vm) = (v, v);
            vp[k] += h;
            vm[k] -= h;
            (res(vp)[i] - res(vm)[i]) / (2.0 * h)
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
        let r = res(v);
        let Some(inv) = jacobian(v).try_inverse() else {
            break;
        };
        let step = -(&inv * &r);
        // converged: the full Newton step is below BB_TOL relative to the variable plus its typical size
        if (0..2).all(|k| step[k].abs() <= BB_TOL * (v[k].abs() + scale[k])) {
            v = [v[0] + step[0], v[1] + step[1]];
            converged = true;
            break;
        }
        let mut t = 1.0;
        loop {
            let trial = [v[0] + t * step[0], v[1] + t * step[1]];
            let valid = trial[0] > 0.0;
            if (valid && norm(&res(trial)) <= norm(&r)) || t < 1e-6 {
                if valid {
                    v = trial;
                }
                break;
            }
            t *= 0.5;
        }
    }
    let (true, Some(inv)) = (converged, jacobian(v).try_inverse()) else {
        let nan = D::from(f64::NAN);
        return (nan, nan, DVector::from_element(n, nan));
    };
    let mut w = [D::from(v[0]), D::from(v[1])];
    for _ in 0..EXTRA_ITERATIONS {
        let r = bernard_blum(w, rho, z, d, delta, lb, &pre).residuals;
        let step: Vec<D> = (0..2)
            .map(|i| (0..2).fold(D::zero(), |acc, k| acc + r[k] * inv[(i, k)]))
            .collect();
        w = [w[0] - step[0], w[1] - step[1]];
    }
    let bb = bernard_blum(w, rho, z, d, delta, lb, &pre);
    let a = bernard_blum_helmholtz(w, rho, z, d, lb, &bb);
    (a, w[0], bb.x)
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
        assert_relative_eq!(dm.ln_delta[(0, 1)].exp(), k, max_relative = 1e-4);
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
            assert_relative_eq!(x[0], x0[0].exp(), max_relative = 1e-11);
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
