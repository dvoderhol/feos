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
//!
//! The association constants, the strengths Δ_ij and the fractions X_i are evaluated as logarithms where their
//! values exceed the range of f64 (large Bjerrum lengths at gas-like densities, where the relative permittivity
//! approaches 1).
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::{LN_2, PI};

/// Euler-Mascheroni constant
#[expect(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// Maximum number of terms of the series of the exponential integral.
const EI_MAX_TERMS: usize = 400;
const MAX_ITER: usize = 200;
/// Tolerance of ||g||₂ in [non_paired_fractions] (default tolerance of the association term of feos).
const TOL: f64 = 1e-10;
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
/// K / (L/mol) = K / Å³ × N_A × 1e-27. The difference of G loses precision for l/σ above about 250 and e^(l/σ)
/// exceeds the range of f64 above about 709; the model uses [ln_bjerrum_constant].
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
/// with l = |z_i z_j| λ_B (Å) and the binding distance factor θ (> 1); positive for all l > 0. Range of l/σ as for
/// [bjerrum_constant]; the model uses [ln_shell_constant].
pub fn shell_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D, theta: D) -> D {
    l.powi(3) * (bjerrum_primitive(l / sigma) - bjerrum_primitive(l / (sigma * theta))) * (4.0 * PI)
}

/// Ebeling association constant of an ion pair (Å³, per pair, number-density based; Ebeling, Z. Phys. Chem.
/// (Leipzig) 238 (1968) 400):
///
/// K = 8π σ³ Σ_{m≥2} b^{2m} / ((2m)! (2m - 3)),   b = l / σ,
///
/// with l = |z_i z_j| λ_B (Å) and the distance of closest approach σ (Å). K is analytic in b (no cutoff), of order
/// b⁴ for small b and equal to the Bjerrum constant to leading order for large b. The series has at most 400 terms
/// (b up to about 700); the model uses [ln_ebeling_constant].
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

/// t from which [scaled_primitive] is evaluated from its asymptotic series.
const ASYMPTOTIC_LIMIT: f64 = 60.0;

/// Scaled primitive H(t) = G(t) t⁴ e^(-t) of [bjerrum_primitive]: from G for t < 60, otherwise from the asymptotic
/// series H(t) = Σ_{k≥0} (k+3)! / (6 t^k), truncated where the real part of a term is below 1e-17 of the sum (the
/// terms decrease for k < t - 4; for t >= 60 the smallest term is below 1e-20 of the sum).
fn scaled_primitive<D: DualNum<Primitive = f64> + Copy>(t: D) -> D {
    if t.re() < ASYMPTOTIC_LIMIT {
        return bjerrum_primitive(t) * t.powi(4) * (-t).exp();
    }
    let mut term = D::one();
    let mut sum = D::one();
    for k in 1..EI_MAX_TERMS {
        term = term * ((k + 3) as f64) / t;
        sum += term;
        if term.re().abs() < 1e-17 * sum.re().abs() {
            break;
        }
    }
    sum
}

/// ln(4π l³ [G(t1) - G(t2)]) for t1 > t2 > 0 and t1 >= 60:
/// ln(4π l³) + t1 - 4 ln t1 + ln[H(t1) - (t1/t2)⁴ e^(t2 - t1) H(t2)] with H of [scaled_primitive] (no overflow of
/// e^t1).
fn ln_primitive_difference<D: DualNum<Primitive = f64> + Copy>(l: D, t1: D, t2: D) -> D {
    let bracket = scaled_primitive(t1) - (t1 / t2).powi(4) * (t2 - t1).exp() * scaled_primitive(t2);
    (l.powi(3) * (4.0 * PI)).ln() + t1 - t1.ln() * 4.0 + bracket.ln()
}

/// ln K of [bjerrum_constant] (None for l / σ <= 2, K = 0): the logarithm of [bjerrum_constant] for l / σ < 60,
/// otherwise from [ln_primitive_difference].
pub fn ln_bjerrum_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D) -> Option<D> {
    let b = l / sigma;
    if b.re() <= 2.0 {
        None
    } else if b.re() < ASYMPTOTIC_LIMIT {
        Some(bjerrum_constant(l, sigma).ln())
    } else {
        Some(ln_primitive_difference(l, b, D::from(2.0)))
    }
}

/// ln K of [shell_constant]: the logarithm of [shell_constant] for l / σ < 60, otherwise from
/// [ln_primitive_difference].
pub fn ln_shell_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D, theta: D) -> D {
    if (l / sigma).re() < ASYMPTOTIC_LIMIT {
        shell_constant(l, sigma, theta).ln()
    } else {
        ln_primitive_difference(l, l / sigma, l / (sigma * theta))
    }
}

/// ln K of [ebeling_constant]: the logarithm of the series for b = l/σ < 60, otherwise
/// ln(4π l³) + b - 4 ln b + ln H(b) ([scaled_primitive]); the terms left out are of relative order b³ e^(-b).
pub fn ln_ebeling_constant<D: DualNum<Primitive = f64> + Copy>(l: D, sigma: D) -> D {
    let b = l / sigma;
    if b.re() < ASYMPTOTIC_LIMIT {
        return ebeling_constant(l, sigma).ln();
    }
    (l.powi(3) * (4.0 * PI)).ln() + b - b.ln() * 4.0 + scaled_primitive(b).ln()
}

/// Association constant of an ion pair used by the ion-pairing variants.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PairingConstant {
    /// Bjerrum constant (`bjerrum_constant`; zero for l / σ <= 2).
    Bjerrum,
    /// Ebeling constant (`ebeling_constant`; analytic in l / σ).
    Ebeling,
    /// Contact-pair constant over the shell σ..θσ (`shell_constant`); the value is the initial binding distance
    /// factor θ of the parameters (`shell_factor` of the model parameters, adjustable).
    Shell(f64),
}

impl Default for PairingConstant {
    /// Contact-pair constant with θ = 1.034 (Naseri Boroujeni et al., J. Chem. Phys. 161 (2024) 057103).
    fn default() -> Self {
        Self::Shell(1.034)
    }
}

impl PairingConstant {
    /// Logarithm of the association constant (Å³) for l = |z_i z_j| λ_B, the distance of closest approach σ and the
    /// binding distance factor `shell_factor` (used by [PairingConstant::Shell] only); None if K = 0.
    pub fn ln_constant<D: DualNum<Primitive = f64> + Copy>(
        &self,
        l: D,
        sigma: D,
        shell_factor: D,
    ) -> Option<D> {
        match self {
            Self::Bjerrum => ln_bjerrum_constant(l, sigma),
            Self::Ebeling => Some(ln_ebeling_constant(l, sigma)),
            Self::Shell(_) => Some(ln_shell_constant(l, sigma, shell_factor)),
        }
    }
}

/// Association strengths of the cation-anion pairs `pairs` (i < j) as logarithms ln Δ_ij (Å³) in `ln_delta`
/// (Δ_ij = Δ_ji; Δ = 0 for all combinations that are not in `pairs`).
#[derive(Clone, Debug)]
pub struct PairingStrengths<D> {
    pub ln_delta: DMatrix<D>,
    pub pairs: Vec<(usize, usize)>,
}

impl<D: DualNum<Primitive = f64> + Copy> PairingStrengths<D> {
    /// The strengths with every ln Δ_ij mapped by `f` (e.g. to the real parts).
    pub fn map<E: DualNum<Primitive = f64> + Copy>(
        &self,
        f: impl Fn(D) -> E,
    ) -> PairingStrengths<E> {
        PairingStrengths {
            ln_delta: self.ln_delta.map(f),
            pairs: self.pairs.clone(),
        }
    }
}

/// ln Δ from which the closed form of [non_paired_fractions] is evaluated in logarithms.
const LN_DELTA_LOG_FORM: f64 = 200.0;
/// ln Δ up to which e^(-ln Δ) is evaluated in the logarithmic closed form for equal densities.
const LN_DELTA_UNDERFLOW: f64 = 700.0;
/// Largest change of ln(ρ_i X_i) per Newton step of [log_fractions].
const MAX_LOG_STEP: f64 = 2.0;
/// Largest number of step halvings per Newton step of [log_fractions].
const MAX_BACKTRACK: usize = 40;

/// ln(1 + e^z)
fn softplus<D: DualNum<Primitive = f64> + Copy>(z: D) -> D {
    if z.re() > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Fractions of non-paired ions X_i and their logarithms ln X_i for the densities `rho` (Å⁻³) and the association
/// strengths `strengths`, solved as in the association term of feos ([crate::association]):
///
/// - one cation-anion pair: closed form of the association of one A and one B site ([closed_form]), evaluated in the
///   number type D;
/// - several pairs: Newton iteration of Michelsen ([michelsen]) if all ln Δ_ij < LN_DELTA_LOG_FORM; if it does not
///   converge, or for larger ln Δ_ij, Newton iteration for the logarithms ln X_i ([log_fractions]). NaN if neither
///   converges.
pub fn non_paired_fractions<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    strengths: &PairingStrengths<D>,
) -> (DVector<D>, DVector<D>) {
    let n = rho.len();
    match strengths.pairs[..] {
        [] => return (DVector::from_element(n, D::one()), DVector::zeros(n)),
        [(a, b)] => {
            let mut ln_x = DVector::zeros(n);
            let (ln_xa, ln_xb) = closed_form(rho[a], rho[b], strengths.ln_delta[(a, b)]);
            ln_x[a] = ln_xa;
            ln_x[b] = ln_xb;
            return (ln_x.map(|v| v.exp()), ln_x);
        }
        _ => (),
    }
    if strengths
        .pairs
        .iter()
        .all(|&(i, j)| strengths.ln_delta[(i, j)].re() < LN_DELTA_LOG_FORM)
        && let Some(x) = michelsen(rho, strengths)
    {
        let ln_x = x.map(|v| v.ln());
        return (x, ln_x);
    }
    let ln_x = log_fractions(rho, strengths);
    (ln_x.map(|v| v.exp()), ln_x)
}

/// X_i of several pairs: Newton iteration of Michelsen (Ind. Eng. Chem. Res. 45 (2006) 8449) for
/// g_i = 1/X_i - 1 - Σ_j ρ_j X_j Δ_ij = 0 in f64 from X_i = 0.2, where a Newton step that would lower X_i by 0.8 X_i or
/// more is replaced by X_i → 0.2 X_i; converged for ||g||₂ < TOL; then Newton steps in the number type D for the
/// derivatives with respect to everything that D carries ([newton]). None if the iteration does not converge.
fn michelsen<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    strengths: &PairingStrengths<D>,
) -> Option<DVector<D>> {
    let n = rho.len();
    let mut delta = DMatrix::zeros(n, n);
    for &(i, j) in &strengths.pairs {
        delta[(i, j)] = strengths.ln_delta[(i, j)].exp();
        delta[(j, i)] = delta[(i, j)];
    }
    let rho_re = rho.map(|r| r.re());
    let delta_re = delta.map(|d: D| d.re());
    let mut x = DVector::from_element(n, 0.2);
    for _ in 0..MAX_ITER {
        let s = DVector::from_fn(n, |i, _| {
            (0..n).fold(0.0, |acc, j| acc + rho_re[j] * x[j] * delta_re[(i, j)])
        });
        let mut g = DVector::from_fn(n, |i, _| 1.0 / x[i] - 1.0 - s[i]);
        if g.norm() < TOL {
            return Some(newton(&x.map(D::from), rho, &delta));
        }
        let h = DMatrix::from_fn(n, n, |i, j| {
            let diag = if i == j { -(1.0 + s[i]) / x[i] } else { 0.0 };
            diag - delta_re[(i, j)] * rho_re[j]
        });
        if !h.lu().solve_mut(&mut g) {
            return None;
        }
        for i in 0..n {
            if g[i] < 0.8 * x[i] {
                x[i] -= g[i];
            } else {
                x[i] *= 0.2;
            }
        }
    }
    None
}

/// (ln X_a, ln X_b) of one cation and one anion with the association strength Δ = e^(ln Δ): the solution of
/// X_a (1 + ρ_b X_b Δ) = 1 and X_b (1 + ρ_a X_a Δ) = 1 from the root of the quadratic equation for the species of the
/// lower density ρ_lo (as for one A and one B site in [crate::association]),
///
/// X_lo = 2 / (q Δ + [(q Δ)² + 4 ρ_lo Δ]^1/2),   q = ρ_hi - ρ_lo + 1/Δ,   X_hi = 1 / (1 + ρ_lo X_lo Δ).
///
/// For ln Δ >= LN_DELTA_LOG_FORM the same expressions are evaluated in logarithms (Δ beyond the range of f64 at
/// gas-like densities, where the Bjerrum length is large): ln X_lo = ln 2 - ln Δ - ln(q + [q² + 4 ρ_lo/Δ]^1/2),
/// ln X_hi = -ln(1 + 2 ρ_lo / (q + [q² + 4 ρ_lo/Δ]^1/2)). In this range, for ρ_lo = 0 (X_hi = 1) the derivatives with
/// respect to ρ_lo, and for equal densities and ln Δ >= LN_DELTA_UNDERFLOW (q = 1/Δ) the derivatives with respect to
/// ρ_hi - ρ_lo, are not included.
fn closed_form<D: DualNum<Primitive = f64> + Copy>(rho_a: D, rho_b: D, ln_delta: D) -> (D, D) {
    let (lo, hi, swap) = if rho_a.re() > rho_b.re() {
        (rho_b, rho_a, false)
    } else {
        (rho_a, rho_b, true)
    };
    let (ln_x_lo, ln_x_hi) = if ln_delta.re() < LN_DELTA_LOG_FORM {
        let delta = ln_delta.exp();
        let excess = delta * (hi - lo) + 1.0;
        let x_lo = ((excess.powi(2) + delta * lo * 4.0).sqrt() + excess).recip() * 2.0;
        let x_hi = (delta * lo * x_lo + 1.0).recip();
        (x_lo.ln(), x_hi.ln())
    } else if lo.re() == 0.0 {
        let ln_x_lo = if hi.re() > 0.0 {
            -softplus(hi.ln() + ln_delta)
        } else {
            D::zero()
        };
        (ln_x_lo, D::zero())
    } else {
        let diff = hi - lo;
        let ln_q = if diff.re() > 0.0 || ln_delta.re() < LN_DELTA_UNDERFLOW {
            (diff + (-ln_delta).exp()).ln()
        } else {
            -ln_delta
        };
        let ln_r = (lo * 4.0).ln() - ln_delta;
        // ln(q + [q² + r]^1/2) without overflow or underflow of q² and r
        let ln_den = if (ln_q * 2.0).re() >= ln_r.re() {
            ln_q + ((ln_r - ln_q * 2.0).exp() + 1.0).sqrt().ln_1p()
        } else {
            ln_r * 0.5 + ((ln_q - ln_r * 0.5).exp() + ((ln_q * 2.0 - ln_r).exp() + 1.0).sqrt()).ln()
        };
        (
            -ln_delta - ln_den + LN_2,
            -softplus((lo * 2.0).ln() - ln_den),
        )
    };
    if swap {
        (ln_x_lo, ln_x_hi)
    } else {
        (ln_x_hi, ln_x_lo)
    }
}

/// Newton steps for g_i(X) = X_i (1 + Σ_j ρ_j X_j Δ_ij) - 1 = 0 in the number type D with the Jacobian of the real
/// parts, started from the converged real solution: every step makes one more order of the derivatives exact
/// (EXTRA_ITERATIONS steps).
fn newton<D: DualNum<Primitive = f64> + Copy>(
    x0: &DVector<D>,
    rho: &DVector<D>,
    delta: &DMatrix<D>,
) -> DVector<D> {
    let n = rho.len();
    let mut x = x0.clone();
    for _ in 0..EXTRA_ITERATIONS {
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

/// ln X_i of several pairs from the mass-action law in logarithms. With λ_i = ln ρ_i + ln X_i (logarithm of the
/// density of non-paired ions i, ρ_i > 0), the free densities y_i = e^λ_i and the pair densities
/// p_ij = e^(λ_i + λ_j + ln Δ_ij), the mass balances are
///
/// F_i = ln(b_i / ρ_i) = ln X_i + ln(1 + Σ_j e^(ln ρ_j + ln X_j + ln Δ_ij)) = 0,   b_i = y_i + Σ_j p_ij,
///
/// with the sum evaluated as softplus of a log-sum-exp. b - ρ is the gradient of the convex function
/// Σ_i (y_i - ρ_i λ_i) + Σ_pairs p_ij; its Hessian H = diag(y) + Bᵀ diag(p) B (B: incidence matrix of pairs and ions)
/// gives the Newton step H Δλ = -(b - ρ), solved with [HessianSolver]. Pairs connect cations and anions only: in every connected group of ions with signs s_i = ±1 (cations and anions), the direction s leaves all
/// p_ij unchanged and E = Σ_i s_i (y_i - ρ_i) = Σ_i s_i (b_i - ρ_i) holds exactly. The component of b - ρ along s is
/// taken as E, which is evaluated without the pair densities: this resolves free densities far below the rounding
/// error of ρ_i when almost all ions are paired.
///
/// Iteration in f64 from ln X_i = -ln(1 + max_j ρ_j Δ_ij) / 2: E = 0 is solved exactly along the signs of every group
/// ([balance_groups]); the Newton step projected onto the orthogonal complement of the sign vectors (a descent direction
/// of the convex function where E = 0) is scaled to max_i |Δλ_i| <= MAX_LOG_STEP and halved (at most MAX_BACKTRACK
/// times) until, after [balance_groups], the convex function decreases by at least 1e-4 of its linear prediction or
/// max_i |F_i| decreases; converged for max_i |F_i| < TOL and |E| <= TOL (Σ_i y_i + |Σ_i s_i ρ_i|) in every group;
/// then EXTRA_ITERATIONS Newton steps in the number type D. Ions with ρ_i = 0: ln X_i = -ln(1 + Σ_j ρ_j X_j Δ_ij) from
/// the solution of the others; the derivatives with respect to their densities are not included. NaN if the iteration
/// does not converge.
fn log_fractions<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    strengths: &PairingStrengths<D>,
) -> DVector<D> {
    let n = rho.len();
    let present: Vec<bool> = rho.iter().map(|r| r.re() > 0.0).collect();
    let pairs: Vec<(usize, usize, D)> = strengths
        .pairs
        .iter()
        .filter(|&&(i, j)| present[i] && present[j])
        .map(|&(i, j)| (i, j, strengths.ln_delta[(i, j)]))
        .collect();
    let groups = signed_groups(n, &pairs);
    let ln_rho = DVector::from_fn(n, |i, _| if present[i] { rho[i].ln() } else { D::zero() });
    let nan = || DVector::from_element(n, D::from(f64::NAN));

    let pairs_re: Vec<(usize, usize, f64)> =
        pairs.iter().map(|&(i, j, l)| (i, j, l.re())).collect();
    let ln_rho_re = ln_rho.map(|v| v.re());
    let rho_re = rho.map(|r| r.re());
    let relative_invariant = |ln_x: &DVector<f64>| {
        groups.iter().fold(0.0, |acc: f64, group| {
            let e = invariant(&rho_re, &ln_rho_re, group, ln_x);
            let free: f64 = group
                .iter()
                .map(|&(i, _)| (ln_rho_re[i] + ln_x[i]).exp())
                .sum();
            acc.max(e.abs() / (free + group_excess(&rho_re, group).abs()))
        })
    };
    let mut ln_x = DVector::zeros(n);
    for &(i, j, ln_delta) in &pairs_re {
        ln_x[i] = f64::min(ln_x[i], -0.5 * softplus(ln_rho_re[j] + ln_delta));
        ln_x[j] = f64::min(ln_x[j], -0.5 * softplus(ln_rho_re[i] + ln_delta));
    }
    balance_groups(&rho_re, &ln_rho_re, &groups, &mut ln_x);
    let mut f = log_balance(&ln_rho_re, &pairs_re, &ln_x);
    let mut phi = dual_function(&ln_rho_re, &pairs_re, &present, &ln_x);
    let mut converged = false;
    for _ in 0..MAX_ITER {
        if f.amax() < TOL && relative_invariant(&ln_x) <= TOL {
            converged = true;
            break;
        }
        let Some(hessian) = HessianSolver::new(&ln_rho_re, &pairs_re, &present, &groups, &ln_x)
        else {
            break;
        };
        let (gradient, r_v) = gradient(&rho_re, &ln_rho_re, &present, &groups, &f, &ln_x);
        let (_, beta) = hessian.solve(&r_v, &gradient);
        let mut step = -(&hessian.u * beta);
        let max = step.amax();
        if !max.is_finite() {
            break;
        }
        if max > MAX_LOG_STEP {
            step *= MAX_LOG_STEP / max;
        }
        let mut slope = gradient.dot(&step);
        let mut accepted = false;
        for _ in 0..MAX_BACKTRACK {
            let mut trial = &ln_x + &step;
            balance_groups(&rho_re, &ln_rho_re, &groups, &mut trial);
            let f_trial = log_balance(&ln_rho_re, &pairs_re, &trial);
            let phi_trial = dual_function(&ln_rho_re, &pairs_re, &present, &trial);
            if phi_trial <= phi + 1e-4 * slope || f_trial.amax() < f.amax() {
                (ln_x, f, phi) = (trial, f_trial, phi_trial);
                accepted = true;
                break;
            }
            step *= 0.5;
            slope *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    if !converged {
        return nan();
    }
    let mut ln_x_d = ln_x.map(D::from);
    for _ in 0..EXTRA_ITERATIONS {
        let ln_x_re = ln_x_d.map(|v| v.re());
        let Some(hessian) = HessianSolver::new(&ln_rho_re, &pairs_re, &present, &groups, &ln_x_re)
        else {
            return nan();
        };
        let f = log_balance(&ln_rho, &pairs, &ln_x_d);
        let (gradient, r_v) = gradient(rho, &ln_rho, &present, &groups, &f, &ln_x_d);
        let (alpha, beta) = hessian.solve(&r_v, &gradient);
        ln_x_d -= mat_vec(&hessian.v, &alpha) + mat_vec(&hessian.u, &beta);
    }
    let mut absent_terms: Vec<Vec<D>> = vec![Vec::new(); n];
    for &(i, j) in &strengths.pairs {
        for (a, c) in [(i, j), (j, i)] {
            if !present[a] && present[c] {
                absent_terms[a].push(ln_rho[c] + ln_x_d[c] + strengths.ln_delta[(i, j)]);
            }
        }
    }
    for (a, terms) in absent_terms.iter().enumerate() {
        if let Some(max) = terms.iter().map(|t| t.re()).reduce(f64::max) {
            let lse = terms
                .iter()
                .fold(D::zero(), |acc, &t| acc + (t - max).exp())
                .ln()
                + max;
            ln_x_d[a] = -softplus(lse);
        }
    }
    ln_x_d
}

/// Connected groups of the ions of `pairs` with the signs s_i = ±1 of a two-coloring (cations and anions); empty if
/// the pairs are not bipartite.
fn signed_groups<D>(n: usize, pairs: &[(usize, usize, D)]) -> Vec<Vec<(usize, f64)>> {
    let mut neighbors = vec![Vec::new(); n];
    for &(i, j, _) in pairs {
        neighbors[i].push(j);
        neighbors[j].push(i);
    }
    let mut sign = vec![0.0; n];
    let mut groups = Vec::new();
    for start in 0..n {
        if sign[start] != 0.0 || neighbors[start].is_empty() {
            continue;
        }
        sign[start] = 1.0;
        let mut group = vec![(start, 1.0)];
        let mut k = 0;
        while k < group.len() {
            let (i, s) = group[k];
            for &j in &neighbors[i] {
                if sign[j] == 0.0 {
                    sign[j] = -s;
                    group.push((j, -s));
                } else if sign[j] == s {
                    return Vec::new();
                }
            }
            k += 1;
        }
        groups.push(group);
    }
    groups
}

/// Σ_i s_i ρ_i of a group as (Σ_(s_i = 1) ρ_i) - (Σ_(s_i = -1) ρ_i), with one rounding in the difference.
fn group_excess<D: DualNum<Primitive = f64> + Copy>(rho: &DVector<D>, group: &[(usize, f64)]) -> D {
    let sum = |sign: f64| {
        group
            .iter()
            .filter(|&&(_, s)| s == sign)
            .fold(D::zero(), |acc, &(i, _)| acc + rho[i])
    };
    sum(1.0) - sum(-1.0)
}

/// E = Σ_i s_i y_i - Σ_i s_i ρ_i of a signed group of [log_fractions] at ln X (the sums separately, so that the
/// rounding error of ρ_i enters only through the constant [group_excess]).
fn invariant<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    ln_rho: &DVector<D>,
    group: &[(usize, f64)],
    ln_x: &DVector<D>,
) -> D {
    let free = group.iter().fold(D::zero(), |acc, &(i, s)| {
        acc + (ln_rho[i] + ln_x[i]).exp() * s
    });
    free - group_excess(rho, group)
}

/// Exact solution of E = 0 along the signs of every group of [log_fractions]: ln X_i → ln X_i + s_i a leaves all pair
/// densities unchanged and gives E(a) = A e^a - B e^(-a) - c with A = Σ_(s_i = 1) y_i, B = Σ_(s_i = -1) y_i and the
/// excess c = Σ_i s_i ρ_i, so that w = e^a is the positive root of A w² - c w - B = 0 (evaluated in logarithms).
fn balance_groups(
    rho: &DVector<f64>,
    ln_rho: &DVector<f64>,
    groups: &[Vec<(usize, f64)>],
    ln_x: &mut DVector<f64>,
) {
    let ln_sum = |terms: Vec<f64>| {
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
    };
    for group in groups {
        let lambda = |sign: f64| {
            group
                .iter()
                .filter(|&&(_, s)| s == sign)
                .map(|&(i, _)| ln_rho[i] + ln_x[i])
                .collect::<Vec<_>>()
        };
        let (ln_a, ln_b) = (ln_sum(lambda(1.0)), ln_sum(lambda(-1.0)));
        let excess = group_excess(rho, group);
        let a = if excess == 0.0 {
            0.5 * (ln_b - ln_a)
        } else {
            // r = 4 A B / c², [c² + 4 A B]^1/2 = |c| (1 + r)^1/2
            let ln_r = 4.0f64.ln() + ln_a + ln_b - 2.0 * excess.abs().ln();
            let ln_1p_sqrt = (0.5 * softplus(ln_r)).exp().ln_1p();
            if excess > 0.0 {
                excess.ln() + ln_1p_sqrt - LN_2 - ln_a
            } else {
                LN_2 + ln_b - (-excess).ln() - ln_1p_sqrt
            }
        };
        for &(i, s) in group {
            ln_x[i] += s * a;
        }
    }
}

/// Gradient b - ρ = ρ (e^F - 1) of the convex function of [log_fractions] (zero for ions with ρ = 0) and its
/// components along the normalized sign vectors of the groups, E / n_group^1/2 with the exact invariants E.
fn gradient<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    ln_rho: &DVector<D>,
    present: &[bool],
    groups: &[Vec<(usize, f64)>],
    f: &DVector<D>,
    ln_x: &DVector<D>,
) -> (DVector<D>, DVector<D>) {
    let g = DVector::from_fn(rho.len(), |i, _| {
        if present[i] {
            rho[i] * f[i].exp_m1()
        } else {
            D::zero()
        }
    });
    let r_v = DVector::from_fn(groups.len(), |k, _| {
        invariant(rho, ln_rho, &groups[k], ln_x) / (groups[k].len() as f64).sqrt()
    });
    (g, r_v)
}

/// F_i = ln X_i + ln(1 + Σ_j e^(ln ρ_j + ln X_j + ln Δ_ij)) of [log_fractions] for the pairs (i, j, ln Δ_ij) of ions
/// with ρ > 0; F_i = 0 for ions without pairs.
fn log_balance<D: DualNum<Primitive = f64> + Copy>(
    ln_rho: &DVector<D>,
    pairs: &[(usize, usize, D)],
    ln_x: &DVector<D>,
) -> DVector<D> {
    let n = ln_rho.len();
    let mut terms: Vec<Vec<D>> = vec![Vec::new(); n];
    for &(i, j, ln_delta) in pairs {
        terms[i].push(ln_rho[j] + ln_x[j] + ln_delta);
        terms[j].push(ln_rho[i] + ln_x[i] + ln_delta);
    }
    DVector::from_fn(n, |i, _| {
        let Some(max) = terms[i].iter().map(|t| t.re()).reduce(f64::max) else {
            return D::zero();
        };
        let lse = terms[i]
            .iter()
            .fold(D::zero(), |acc, &t| acc + (t - max).exp())
            .ln()
            + max;
        ln_x[i] + softplus(lse)
    })
}

/// Dual function Σ_i (y_i - ρ_i λ_i) + Σ_pairs p_ij of [log_fractions] at ln X (λ_i = ln ρ_i + ln X_i, ions with ρ > 0).
fn dual_function(
    ln_rho: &DVector<f64>,
    pairs: &[(usize, usize, f64)],
    present: &[bool],
    ln_x: &DVector<f64>,
) -> f64 {
    let lambda = ln_rho + ln_x;
    let free: f64 = (0..ln_rho.len())
        .filter(|&i| present[i])
        .map(|i| lambda[i].exp() - ln_rho[i].exp() * lambda[i])
        .sum();
    let paired: f64 = pairs
        .iter()
        .map(|&(i, j, ln_delta)| (lambda[i] + lambda[j] + ln_delta).exp())
        .sum();
    free + paired
}

/// Solution of H d = g for the Hessian H = diag(y) + Bᵀ diag(p) B of [log_fractions] at ln X (ions with ρ = 0:
/// H_ii = 1 without pairs). B v = 0 for the sign vectors v of the groups, so H v = diag(y) v. With the normalized sign
/// vectors as columns of V and an orthonormal complement U, d = V α + U β from the blocks
/// [Vᵀ diag(y) V, Cᵀ; C, M] [α; β] = [Vᵀ g; Uᵀ g] with C = Uᵀ diag(y) V and M = Uᵀ H U (Schur complement in α),
/// where the blocks with V are evaluated from y directly instead of from H, and Vᵀ g is given separately (exact
/// invariants E instead of the rounded components of g).
struct HessianSolver {
    v: DMatrix<f64>,
    u: DMatrix<f64>,
    c: DMatrix<f64>,
    m_inv: DMatrix<f64>,
    schur_inv: DMatrix<f64>,
}

impl HessianSolver {
    fn new(
        ln_rho: &DVector<f64>,
        pairs: &[(usize, usize, f64)],
        present: &[bool],
        groups: &[Vec<(usize, f64)>],
        ln_x: &DVector<f64>,
    ) -> Option<Self> {
        let n = ln_rho.len();
        let y = DVector::from_fn(n, |i, _| {
            if present[i] {
                (ln_rho[i] + ln_x[i]).exp()
            } else {
                1.0
            }
        });
        let mut h = DMatrix::from_diagonal(&y);
        for &(i, j, ln_delta) in pairs {
            let p = (ln_rho[i] + ln_x[i] + ln_rho[j] + ln_x[j] + ln_delta).exp();
            h[(i, i)] += p;
            h[(j, j)] += p;
            h[(i, j)] += p;
            h[(j, i)] += p;
        }
        let mut v = DMatrix::zeros(n, groups.len());
        for (k, group) in groups.iter().enumerate() {
            let norm = (group.len() as f64).sqrt();
            for &(i, s) in group {
                v[(i, k)] = s / norm;
            }
        }
        // orthonormal complement of V: Gram-Schmidt (twice) of the unit vectors
        let mut basis: Vec<DVector<f64>> = v.column_iter().map(|c| c.into_owned()).collect();
        let mut u_columns = Vec::new();
        for i in 0..n {
            let mut e = DVector::zeros(n);
            e[i] = 1.0;
            for _ in 0..2 {
                for b in &basis {
                    let projection = b.dot(&e);
                    e.axpy(-projection, b, 1.0);
                }
            }
            let norm = e.norm();
            if norm > 1e-8 {
                e /= norm;
                basis.push(e.clone());
                u_columns.push(e);
            }
        }
        if u_columns.len() + groups.len() != n {
            return None;
        }
        let u = DMatrix::from_columns(&u_columns);
        let yv = DMatrix::from_fn(n, groups.len(), |i, k| y[i] * v[(i, k)]);
        let c = u.transpose() * &yv;
        let m_inv = (u.transpose() * &h * &u).try_inverse()?;
        let schur = v.transpose() * &yv - c.transpose() * &m_inv * &c;
        let schur_inv = schur.try_inverse()?;
        let finite = |m: &DMatrix<f64>| m.iter().all(|x| x.is_finite());
        (finite(&m_inv) && finite(&schur_inv)).then_some(Self {
            v,
            u,
            c,
            m_inv,
            schur_inv,
        })
    }

    /// (α, β) of H (V α + U β) = g for the components r_v = Vᵀ g (given separately) and Uᵀ g of the right-hand side.
    fn solve<D: DualNum<Primitive = f64> + Copy>(
        &self,
        r_v: &DVector<D>,
        g: &DVector<D>,
    ) -> (DVector<D>, DVector<D>) {
        let t = mat_vec(&self.m_inv, &mat_vec(&self.u.transpose(), g));
        let alpha = mat_vec(&self.schur_inv, &(r_v - mat_vec(&self.c.transpose(), &t)));
        let beta = t - mat_vec(&self.m_inv, &mat_vec(&self.c, &alpha));
        (alpha, beta)
    }
}

/// Product of a real matrix and a vector of the number type D.
fn mat_vec<D: DualNum<Primitive = f64> + Copy>(a: &DMatrix<f64>, x: &DVector<D>) -> DVector<D> {
    DVector::from_fn(a.nrows(), |i, _| {
        (0..a.ncols()).fold(D::zero(), |acc, j| acc + x[j] * a[(i, j)])
    })
}

/// Density of pairs ρ_i ρ_j X_i X_j Δ_ij from ln(X_i X_j Δ_ij) = `ln_xxd`: as a product of the densities if one of
/// them is zero or ln(X_i X_j Δ_ij) < LN_DELTA_LOG_FORM, otherwise as exp(ln ρ_i + ln ρ_j + ln(X_i X_j Δ_ij)).
pub fn pair_density<D: DualNum<Primitive = f64> + Copy>(rho_i: D, rho_j: D, ln_xxd: D) -> D {
    if rho_i.re() == 0.0 || rho_j.re() == 0.0 || ln_xxd.re() < LN_DELTA_LOG_FORM {
        rho_i * rho_j * ln_xxd.exp()
    } else {
        (rho_i.ln() + rho_j.ln() + ln_xxd).exp()
    }
}

/// Reduced Helmholtz energy density of the pairing a / (kT V) = Σ_i ρ_i (ln X_i - X_i / 2 + 1/2) for the fractions
/// `x` and their logarithms `ln_x`.
pub fn pairing_helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
    rho: &DVector<D>,
    x: &DVector<D>,
    ln_x: &DVector<D>,
) -> D {
    (0..rho.len()).fold(D::zero(), |acc, i| {
        acc + rho[i] * (ln_x[i] - x[i] * 0.5 + 0.5)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;
    use nalgebra::dvector;
    use num_dual::{Dual64, first_derivative};

    /// X of [non_paired_fractions] for a matrix of association strengths Δ (pairs: the entries i < j with Δ_ij != 0).
    fn npf<D: DualNum<Primitive = f64> + Copy>(rho: &DVector<D>, delta: &DMatrix<D>) -> DVector<D> {
        let n = rho.len();
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| delta[(i, j)].re() != 0.0)
            .collect();
        let strengths = PairingStrengths {
            ln_delta: delta.map(|d| if d.re() > 0.0 { d.ln() } else { D::zero() }),
            pairs,
        };
        non_paired_fractions(rho, &strengths).0
    }

    /// ln K_shell / 4π = ln(σ³/t) + t + ln ∫_0^W e^(-w) (1 - w/t)^(-4) dw, W = t (1 - 1/θ), t = l/σ (substitution
    /// w = t (1 - σ/r)), from the composite Simpson rule.
    fn ln_shell_reference(l: f64, sigma: f64, theta: f64) -> f64 {
        let t = l / sigma;
        let w_max = t * (1.0 - 1.0 / theta);
        let n = 200_000;
        let h = w_max / n as f64;
        let f = |w: f64| (-w).exp() * (1.0 - w / t).powi(-4);
        let mut s = f(0.0) + f(w_max);
        for k in 1..n {
            s += f(k as f64 * h) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        (4.0 * PI * sigma.powi(3) / t * s * h / 3.0).ln() + t
    }

    /// ln K of the shell and Bjerrum constants against quadrature from liquid (l/σ = 3) to gas-like (l/σ = 3000)
    /// Bjerrum lengths; equal to the logarithm of [shell_constant] below l/σ = 60.
    #[test]
    fn ln_constants_wide_range() {
        let sigma = 3.0;
        for t in [
            3.0, 10.0, 40.0, 59.0, 61.0, 100.0, 300.0, 709.0, 1000.0, 3000.0,
        ] {
            let l = t * sigma;
            for theta in [1.034, 1.3, 2.1] {
                assert_relative_eq!(
                    ln_shell_constant(l, sigma, theta),
                    ln_shell_reference(l, sigma, theta),
                    epsilon = 1e-9
                );
            }
            // Bjerrum: upper limit l/2, i.e. θ = t/2
            assert_relative_eq!(
                ln_bjerrum_constant(l, sigma).unwrap(),
                ln_shell_reference(l, sigma, t / 2.0),
                epsilon = 1e-9
            );
            if t < ASYMPTOTIC_LIMIT {
                assert_eq!(
                    ln_shell_constant(l, sigma, 1.034),
                    shell_constant(l, sigma, 1.034).ln()
                );
            }
        }
        assert!(ln_bjerrum_constant(5.0, 3.0).is_none());
        // Ebeling: asymptotic form against the series
        for b in [60.0, 100.0, 300.0] {
            assert_relative_eq!(
                ln_ebeling_constant(b, 1.0),
                ebeling_constant(b, 1.0).ln(),
                epsilon = 1e-12
            );
        }
        // derivative with respect to θ of the asymptotic form: dK/dθ = 4π (θσ)² σ exp(l / (θσ))
        let (l, theta) = (300.0 * sigma, 1.034);
        let (_, d) = first_derivative(
            |x: Dual64| ln_shell_constant(Dual64::from(l), Dual64::from(sigma), x),
            theta,
        );
        let r = theta * sigma;
        let ln_dk = (4.0 * PI * r * r * sigma).ln() + l / r;
        assert_relative_eq!(
            d,
            (ln_dk - ln_shell_constant(l, sigma, theta)).exp(),
            max_relative = 1e-10
        );
    }

    /// Mass action law in logarithms, ln X_i + ln(1 + ρ_j X_j Δ) = 0, for Δ beyond the range of f64 (ln Δ up to 3000),
    /// for unequal, equal and one zero density.
    #[test]
    fn closed_form_large_strengths() {
        let residual = |ra: f64, rb: f64, ln_d: f64, (la, lb): (f64, f64)| {
            let ra_term = if rb > 0.0 {
                softplus(rb.ln() + lb + ln_d)
            } else {
                0.0
            };
            let rb_term = if ra > 0.0 {
                softplus(ra.ln() + la + ln_d)
            } else {
                0.0
            };
            (la + ra_term).abs().max((lb + rb_term).abs())
        };
        for ln_d in [150.0, 199.999, 200.001, 400.0, 699.0, 701.0, 1500.0, 3000.0] {
            for (ra, rb) in [
                (1e-3, 2e-3),
                (2e-3, 1e-3),
                (1e-3, 1e-3),
                (1e-7, 1e-7 * (1.0 + 1e-6)),
                (0.0, 1e-3),
            ] {
                let ln_x: (f64, f64) = closed_form(ra, rb, ln_d);
                assert!(
                    ln_x.0.is_finite() && ln_x.1.is_finite(),
                    "{ln_d} {ra} {rb}: {ln_x:?}"
                );
                assert!(ln_x.0 <= 0.0 && ln_x.1 <= 0.0);
                assert!(
                    residual(ra, rb, ln_d, ln_x) < 1e-12 * ln_d,
                    "{ln_d} {ra} {rb}: {ln_x:?}"
                );
            }
        }
        // continuity at the change to the logarithmic form
        let below = closed_form(1e-3, 3e-3, LN_DELTA_LOG_FORM - 1e-9);
        let above = closed_form(1e-3, 3e-3, LN_DELTA_LOG_FORM + 1e-9);
        assert_relative_eq!(below.0, above.0, max_relative = 1e-10);
        assert_relative_eq!(below.1, above.1, max_relative = 1e-10);
    }

    /// Derivatives of the logarithmic closed form with respect to a density (unequal densities; for equal densities
    /// d ln X_a / dρ_a = (Δ/ρ)^1/2 / 2 varies on the scale of the free-ion density (Δ/ρ)^-1/2) and to ln Δ against
    /// central differences.
    #[test]
    fn closed_form_large_strengths_derivative() {
        for (ra, rb, ln_d) in [
            (1e-3, 3e-3, 400.0),
            (2e-6, 1e-6, 900.0),
            (1e-3, 1e-3, 300.0),
        ] {
            if ra != rb {
                let (_, d_rho) = first_derivative(
                    |r: Dual64| closed_form(r, Dual64::from(rb), Dual64::from(ln_d)).0,
                    ra,
                );
                let h = 1e-6 * ra;
                let fd =
                    (closed_form(ra + h, rb, ln_d).0 - closed_form(ra - h, rb, ln_d).0) / (2.0 * h);
                assert_relative_eq!(d_rho, fd, max_relative = 1e-6);
            }
            let (_, d_ln) = first_derivative(
                |v: Dual64| closed_form(Dual64::from(ra), Dual64::from(rb), v).1,
                ln_d,
            );
            let fd =
                (closed_form(ra, rb, ln_d + 1e-5).1 - closed_form(ra, rb, ln_d - 1e-5).1) / 2e-5;
            assert_relative_eq!(d_ln, fd, max_relative = 1e-6, epsilon = 1e-12);
        }
    }

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
            let x = npf(&rho, &delta);
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
            // dK/dθ = 4π (θσ)² σ exp(l / (θσ))
            let (_, dk) = first_derivative(
                |t: Dual64| shell_constant(Dual64::from(l), Dual64::from(sigma), t),
                theta,
            );
            let r = theta * sigma;
            assert_relative_eq!(
                dk,
                4.0 * PI * r * r * sigma * (l / r).exp(),
                max_relative = 1e-10
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
        let x = npf(&dvector![rp, rm], &dmatrix![0.0, d; d, 0.0]);
        let pair = rp * rm * x[0] * x[1] * d;
        let rt = rp + rm;
        let closed = 2.0 * d * rp * rm
            / (1.0 + d * rt + (1.0 + 2.0 * d * rt + d * d * (rp - rm).powi(2)).sqrt());
        assert_relative_eq!(pair, closed, max_relative = 1e-12);
        // material balance: free + paired = total
        assert_relative_eq!(rp * x[0] + pair, rp, max_relative = 1e-12);
        assert_relative_eq!(rm * x[1] + pair, rm, max_relative = 1e-12);
    }

    /// Relative residuals |X_i (1 + Σ_j ρ_j X_j Δ_ij) - 1| of the mass action law.
    fn mass_action_residual(rho: &DVector<f64>, delta: &DMatrix<f64>, x: &DVector<f64>) -> f64 {
        let n = rho.len();
        (0..n)
            .map(|i| {
                let s: f64 = (0..n).map(|j| rho[j] * x[j] * delta[(i, j)]).sum();
                (x[i] * (1.0 + s) - 1.0).abs()
            })
            .fold(0.0, f64::max)
    }

    /// One cation and one anion over ρΔ = 1e-6..1e10 and density ratios 1..1e3: finite, 0 < X <= 1, mass action law.
    #[test]
    fn single_salt_wide_range() {
        for k in [1e-4, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12] {
            for ratio in [1.0, 1.0 + 1e-9, 1.01, 2.0, 1e3] {
                let rho = dvector![1e-2, 1e-2 / ratio];
                let delta = dmatrix![0.0, k; k, 0.0];
                let x = npf(&rho, &delta);
                assert!(x.iter().all(|&x| x > 0.0 && x <= 1.0), "{k} {ratio}: {x}");
                assert!(
                    mass_action_residual(&rho, &delta, &x) < 1e-12,
                    "{k} {ratio}: {x}"
                );
            }
        }
    }

    /// The Newton iteration (forced by a third ion of zero density) gives the closed form of one cation and one anion.
    #[test]
    fn newton_equals_closed_form() {
        for (k, rp, rm) in [
            (150.0, 0.004, 0.002),
            (1e5, 0.003, 0.003),
            (1e7, 0.01, 0.002),
        ] {
            let x2 = npf(&dvector![rp, rm], &dmatrix![0.0, k; k, 0.0]);
            let x3 = npf(
                &dvector![rp, rm, 0.0],
                &dmatrix![0.0, k, 0.0; k, 0.0, 2.0 * k; 0.0, 2.0 * k, 0.0],
            );
            assert_relative_eq!(x2[0], x3[0], max_relative = 1e-9);
            assert_relative_eq!(x2[1], x3[1], max_relative = 1e-9);
        }
    }

    /// Two cations and one anion (Michelsen iteration) over a wide range of strengths: finite, mass action law and
    /// material balance of the anion.
    #[test]
    fn mixture_wide_range() {
        for k in [1e-2, 1e1, 1e3, 1e5, 1e7] {
            for (r1, r2, r3) in [
                (0.004, 0.001, 0.005),
                (0.001, 0.001, 0.0021),
                (0.01, 1e-6, 0.002),
            ] {
                let rho = dvector![r1, r2, r3];
                let delta = dmatrix![0.0, 0.0, k; 0.0, 0.0, 3.0 * k; k, 3.0 * k, 0.0];
                let x = npf(&rho, &delta);
                assert!(x.iter().all(|&x| x > 0.0 && x <= 1.0), "{k}: {x}");
                assert!(mass_action_residual(&rho, &delta, &x) < 1e-9, "{k}: {x}");
                let pairs = r3 * x[2] * (r1 * x[0] * k + r2 * x[1] * 3.0 * k);
                assert_relative_eq!(r3 * x[2] + pairs, r3, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn non_paired_fraction_derivative_mixture() {
        // dX/dρ_1 of the Michelsen iteration (Newton steps in dual numbers) against central differences
        let (r2, r3, k) = (0.001, 0.004, 400.0);
        let delta = dmatrix![0.0, 0.0, k; 0.0, 0.0, 2.0 * k; k, 2.0 * k, 0.0];
        let f = |r1: Dual64| {
            npf(
                &dvector![r1, Dual64::from(r2), Dual64::from(r3)],
                &delta.map(Dual64::from),
            )[2]
        };
        let r1 = 0.002;
        let (_, dx) = first_derivative(f, r1);
        let h = 1e-8;
        let g = |r1: f64| npf(&dvector![r1, r2, r3], &delta)[2];
        assert_relative_eq!(dx, (g(r1 + h) - g(r1 - h)) / (2.0 * h), max_relative = 1e-6);
    }

    #[test]
    fn non_paired_fraction_derivative() {
        // dX/dρ+ from the Newton steps in dual numbers against central differences
        let (rm, d) = (0.003, 400.0);
        let f = |rp: Dual64| {
            npf(
                &dvector![rp, Dual64::from(rm)],
                &dmatrix![Dual64::from(0.0), Dual64::from(d); Dual64::from(d), Dual64::from(0.0)],
            )[0]
        };
        let rp = 0.002;
        let (_, dx) = first_derivative(f, rp);
        let h = 1e-8;
        let g = |rp: f64| npf(&dvector![rp, rm], &dmatrix![0.0, d; d, 0.0])[0];
        assert_relative_eq!(dx, (g(rp + h) - g(rp - h)) / (2.0 * h), max_relative = 1e-6);
    }

    /// strengths of one cation (index 2) paired with two anions: ln Δ_02 = ln_k, ln Δ_12 = ln_k + ln 3
    fn mixture_strengths<D: DualNum<Primitive = f64> + Copy>(ln_k: D) -> PairingStrengths<D> {
        let mut ln_delta = DMatrix::zeros(3, 3);
        ln_delta[(0, 2)] = ln_k;
        ln_delta[(1, 2)] = ln_k + 3.0f64.ln();
        PairingStrengths {
            ln_delta,
            pairs: vec![(0, 2), (1, 2)],
        }
    }

    /// max_i |ln X_i + ln(1 + Σ_j ρ_j X_j Δ_ij)| from ln X and ln Δ, with the sum as log-sum-exp
    fn log_mass_action_residual(
        rho: &DVector<f64>,
        strengths: &PairingStrengths<f64>,
        ln_x: &DVector<f64>,
    ) -> f64 {
        let n = rho.len();
        (0..n)
            .map(|i| {
                let terms: Vec<f64> = strengths
                    .pairs
                    .iter()
                    .filter_map(|&(a, b)| match (a == i, b == i) {
                        (true, _) => Some((b, strengths.ln_delta[(a, b)])),
                        (_, true) => Some((a, strengths.ln_delta[(a, b)])),
                        _ => None,
                    })
                    .map(|(j, ln_delta)| rho[j].ln() + ln_x[j] + ln_delta)
                    .collect();
                let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln();
                (ln_x[i] + softplus(lse)).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn log_fractions_equal_michelsen() {
        let mut compared = 0;
        for ln_k in [-4.0f64, 2.3, 9.2, 16.1, 20.0] {
            for (r1, r2, r3) in [
                (0.004, 0.001, 0.005),
                (0.001, 0.001, 0.0021),
                (0.01, 1e-6, 0.002),
            ] {
                let rho = dvector![r1, r2, r3];
                let strengths = mixture_strengths(ln_k);
                let ln_x = log_fractions(&rho, &strengths);
                if let Some(x) = michelsen(&rho, &strengths) {
                    for i in 0..3 {
                        assert_relative_eq!(ln_x[i].exp(), x[i], max_relative = 1e-9);
                    }
                    compared += 1;
                }
            }
        }
        assert!(compared >= 12, "{compared}");
    }

    #[test]
    fn mixture_large_strengths() {
        for ln_k in [25.0f64, 60.0, 199.0, 201.0, 400.0, 750.0] {
            for (r1, r2, r3) in [
                (0.004, 0.001, 0.005),
                (0.001, 0.001, 0.0021),
                (0.01, 1e-6, 0.002),
                (1e-9, 2e-9, 3e-9),
            ] {
                let rho = dvector![r1, r2, r3];
                let strengths = mixture_strengths(ln_k);
                let (x, ln_x) = non_paired_fractions(&rho, &strengths);
                assert!(x.iter().all(|&x| (0.0..=1.0).contains(&x)), "{ln_k}: {x}");
                assert!(ln_x.iter().all(|v: &f64| v.is_finite()), "{ln_k}: {ln_x}");
                assert!(
                    log_mass_action_residual(&rho, &strengths, &ln_x) < 1e-9,
                    "{ln_k}: {ln_x}"
                );
                // every pair removes one cation and one anion: free cations - free anions = cations - anions
                let free_cations = (rho[2].ln() + ln_x[2]).exp();
                let free_anions = (rho[0].ln() + ln_x[0]).exp() + (rho[1].ln() + ln_x[1]).exp();
                let excess = r3 - (r1 + r2);
                assert_relative_eq!(
                    free_cations - free_anions,
                    excess,
                    max_relative = 1e-9,
                    epsilon = 1e-9 * (free_cations + free_anions)
                );
            }
        }
        // electroneutral (excess exactly zero): ln X_+ + ln X_- = -ln(ρ Δ) for complete pairing
        let rho = dvector![0.004, 0.001, 0.005];
        let ln_x = non_paired_fractions(&rho, &mixture_strengths(400.0)).1;
        assert_relative_eq!(
            ln_x[0] + ln_x[2],
            -(400.0 + 0.005f64.ln()),
            max_relative = 1e-6
        );
        // continuous where the iteration changes from X to ln X
        let rho = dvector![0.004, 0.0011, 0.005];
        let below = non_paired_fractions(&rho, &mixture_strengths(LN_DELTA_LOG_FORM - 1e-9)).1;
        let above = non_paired_fractions(&rho, &mixture_strengths(LN_DELTA_LOG_FORM + 1e-9)).1;
        for i in 0..3 {
            assert_relative_eq!(below[i], above[i], max_relative = 1e-8);
        }
    }

    #[test]
    fn log_fractions_absent_ion() {
        // ion 1 at zero density: the others as one pair (closed form), ln X_1 = -ln(1 + ρ_2 X_2 Δ_12)
        let (r0, r2, ln_k) = (0.003, 0.004, 300.0);
        let rho = dvector![r0, 0.0, r2];
        let strengths = mixture_strengths(ln_k);
        let ln_x = log_fractions(&rho, &strengths);
        let (ln_x0, ln_x2) = closed_form(r0, r2, ln_k);
        assert_relative_eq!(ln_x[0], ln_x0, max_relative = 1e-10);
        assert_relative_eq!(ln_x[2], ln_x2, max_relative = 1e-10);
        let ln_x1 = -softplus(r2.ln() + ln_x2 + ln_k + 3.0f64.ln());
        assert_relative_eq!(ln_x[1], ln_x1, max_relative = 1e-10);
    }

    #[test]
    fn log_fractions_derivative() {
        // d ln X / dρ_0 and d ln X / d ln Δ of the logarithmic iteration against central differences
        let (r1, r2, r3) = (0.002, 0.001, 0.004);
        for ln_k in [250.0f64, 800.0] {
            let f = |r: Dual64| {
                let rho = dvector![r, Dual64::from(r2), Dual64::from(r3)];
                non_paired_fractions(&rho, &mixture_strengths(Dual64::from(ln_k))).1
            };
            let g = |r: f64| non_paired_fractions(&dvector![r, r2, r3], &mixture_strengths(ln_k)).1;
            let h = 1e-9;
            let fd = (g(r1 + h) - g(r1 - h)) / (2.0 * h);
            let d = f(Dual64::new(r1, 1.0)).map(|v| v.eps);
            for i in 0..3 {
                assert_relative_eq!(d[i], fd[i], max_relative = 1e-6, epsilon = 1e-6);
            }
            let rho = dvector![r1, r2, r3];
            let fl = |l: f64| non_paired_fractions(&rho, &mixture_strengths(l)).1;
            let fd = (fl(ln_k + 1e-6) - fl(ln_k - 1e-6)) / 2e-6;
            let d = non_paired_fractions(
                &rho.map(Dual64::from),
                &mixture_strengths(Dual64::new(ln_k, 1.0)),
            )
            .1
            .map(|v| v.eps);
            for i in 0..3 {
                assert_relative_eq!(d[i], fd[i], max_relative = 1e-6, epsilon = 1e-9);
            }
        }
    }
}
