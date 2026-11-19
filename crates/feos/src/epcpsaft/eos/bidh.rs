//! Debye-Hückel term with ion pairing (BiDH; Naseri Boroujeni et al., J. Chem. Phys. 159, 154503, 2023).
//!
//! a / (kT V) = a_DH(ρ, d, λ_B) + Σ_i ρ_i (ln X_i - X_i / 2 + 1/2),
//!
//! with the Debye-Hückel term of all ions (paired or not; [debye_hueckel]) and the pairing of one site per ion
//! ([non_paired_fractions]) with the association strength of a cation i and an anion j
//!
//! Δ_ij = exp(f_ij) K_ij g_ij y_ij,
//!
//! - K_ij: Bjerrum or Ebeling association constant ([PairingConstant]) with l = |z_i z_j| λ_B and the distance of closest
//!   approach d_ij = (d_i + d_j) / 2;
//! - f_ij: logarithm of an adjustable factor (`ion_pairing` of the binary parameters; 0: Bjerrum constant);
//! - g_ij: contact value of the hard-sphere mixture of all species (segments) at d_ij, the same expression as in
//!   the association term;
//! - y_ij = exp(β μ_i + β μ_j - β μ_ij): electrostatic part of the cavity function of the reference without pairs,
//!   with the Debye-Hückel chemical potentials ([debye_hueckel_chemical_potential]) of the ions i and j and of a
//!   test particle of charge z_i + z_j and diameter d_ij at zero density. The Bjerrum length is that of the
//!   solution; a pair has the permittivity contribution of its two ions, so that the composition dependence of the
//!   permittivity does not enter y_ij.
//!
//! The terms of the publication are used without the ion-dipole term of the pairs and in the corrected forms
//! (association term with the sign of Wertheim's theory, Debye-Hückel chemical potential derived from a_DH).
use super::ion_pairing::{PairingConstant, non_paired_fractions, pairing_helmholtz_energy_density};
use super::ionic::{debye_hueckel, debye_hueckel_chemical_potential};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;

/// Reduced Helmholtz energy density of the BiDH term for the ions with densities `rho` (Å⁻³), charges `z`,
/// diameters `d` (Å), contact values `g` of the hard-sphere mixture at (d_i + d_j)/2, logarithms of the pairing
/// factors `f`, Bjerrum length `lb` (Å) and association constant `constant`. Returns (a / (kT V), fractions of
/// non-paired ions X).
pub fn bidh<D: DualNum<Primitive = f64> + Copy>(
    rho: &[D],
    z: &[f64],
    d: &[D],
    g: &DMatrix<D>,
    f: &DMatrix<D>,
    lb: D,
    constant: PairingConstant,
) -> (D, DVector<D>) {
    let n = rho.len();
    let a_dh = debye_hueckel(rho, z, d, lb);
    let delta = pairing_strength(rho, z, d, g, f, lb, constant);
    let rho_v = DVector::from_column_slice(rho);
    if delta.iter().all(|x| x.re() == 0.0) {
        return (a_dh, DVector::from_element(n, D::one()));
    }
    let x = non_paired_fractions(&rho_v, &delta);
    (a_dh + pairing_helmholtz_energy_density(&rho_v, &x), x)
}

/// Association strengths Δ_ij (Å³) of the BiDH term (zero for ions of the same charge sign).
pub fn pairing_strength<D: DualNum<Primitive = f64> + Copy>(
    rho: &[D],
    z: &[f64],
    d: &[D],
    g: &DMatrix<D>,
    f: &DMatrix<D>,
    lb: D,
    constant: PairingConstant,
) -> DMatrix<D> {
    let n = rho.len();
    let mu: Vec<D> = (0..n)
        .map(|i| debye_hueckel_chemical_potential(z[i], d[i], rho, z, d, lb))
        .collect();
    DMatrix::from_fn(n, n, |i, j| {
        if z[i] * z[j] >= 0.0 {
            return D::zero();
        }
        let dij = (d[i] + d[j]) * 0.5;
        let k = constant.constant(lb * (z[i] * z[j]).abs(), dij);
        if k.re() == 0.0 {
            return D::zero();
        }
        let mu_pair = debye_hueckel_chemical_potential(z[i] + z[j], dij, rho, z, d, lb);
        f[(i, j)].exp() * k * g[(i, j)] * (mu[i] + mu[j] - mu_pair).exp()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcpsaft::eos::ion_pairing::bjerrum_constant;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use num_dual::Dual64;

    type Setup = (Vec<f64>, Vec<f64>, Vec<f64>, DMatrix<f64>, DMatrix<f64>);

    fn setup(scale: f64) -> Setup {
        // MgSO4-like 2:2 salt with Na+ and Cl-
        let rho: Vec<f64> = [0.003, 0.003, 0.001, 0.001]
            .iter()
            .map(|r| r * scale)
            .collect();
        let z = vec![2.0, -2.0, 1.0, -1.0];
        let d = vec![2.8, 3.4, 2.6, 2.9];
        let g = DMatrix::from_element(4, 4, 1.2);
        let f = DMatrix::from_element(4, 4, 0.0);
        (rho, z, d, g, f)
    }

    /// Without pairing (Δ = 0 through a large negative factor) the term equals the Debye-Hückel term.
    #[test]
    fn no_pairing_is_debye_hueckel() {
        let (rho, z, d, g, _) = setup(1.0);
        let f = DMatrix::from_element(4, 4, -200.0);
        let (a, x) = bidh(&rho, &z, &d, &g, &f, 7.1, PairingConstant::Bjerrum);
        assert_relative_eq!(a, debye_hueckel(&rho, &z, &d, 7.1), max_relative = 1e-12);
        assert!(x.iter().all(|&x| (x - 1.0).abs() < 1e-12));
    }

    /// At low concentration Δ_ij -> K_ij g y with ln y -> λ_B κ z_i z_j (limiting law for the ions and for the pair
    /// of charge z_i + z_j), i.e. Δ_ij -> K_ij for g = 1 at infinite dilution.
    #[test]
    fn mass_action_at_infinite_dilution() {
        let (rho, z, d, _, f) = setup(1e-12);
        let g = DMatrix::from_element(4, 4, 1.0);
        let lb = 7.1;
        let delta = pairing_strength(&rho, &z, &d, &g, &f, lb, PairingConstant::Bjerrum);
        let k = bjerrum_constant(lb * 4.0, (d[0] + d[1]) * 0.5);
        let kappa = (4.0
            * std::f64::consts::PI
            * lb
            * rho.iter().zip(&z).map(|(r, z)| r * z * z).sum::<f64>())
        .sqrt();
        assert_relative_eq!(
            (delta[(0, 1)] / k).ln(),
            lb * kappa * z[0] * z[1],
            max_relative = 1e-3
        );
        assert_relative_eq!(delta[(0, 1)], k, max_relative = 1e-4);
        // same-sign pairs do not associate
        assert_eq!(delta[(0, 2)], 0.0);
    }

    /// Screening reduces pairing: Δ decreases with the ionic strength.
    #[test]
    fn screening_reduces_pairing() {
        let (_, z, d, g, f) = setup(1.0);
        let lb = 7.1;
        let low =
            pairing_strength(&setup(0.01).0, &z, &d, &g, &f, lb, PairingConstant::Bjerrum)[(0, 1)];
        let high =
            pairing_strength(&setup(1.0).0, &z, &d, &g, &f, lb, PairingConstant::Bjerrum)[(0, 1)];
        assert!(high < low);
    }

    /// Exact derivatives: d(a)/dρ_k from dual numbers equals central differences (the fractions X are solved
    /// inside; by stationarity d a / d ρ_k = ∂a/∂ρ_k at fixed X).
    #[test]
    fn derivative_density() {
        let (rho, z, d, g, f) = setup(1.0);
        let lb = 7.1;
        for k in 0..4 {
            let rho_d: Vec<Dual64> = (0..4)
                .map(|i| Dual64::new(rho[i], if i == k { 1.0 } else { 0.0 }))
                .collect();
            let d_d: Vec<Dual64> = d.iter().map(|&x| Dual64::from(x)).collect();
            let (a, _) = bidh(
                &rho_d,
                &z,
                &d_d,
                &g.map(Dual64::from),
                &f.map(Dual64::from),
                Dual64::from(lb),
                PairingConstant::Bjerrum,
            );
            let h = 1e-7 * rho[k];
            let mut rp = rho.clone();
            let mut rm = rho.clone();
            rp[k] += h;
            rm[k] -= h;
            let fd = (bidh(&rp, &z, &d, &g, &f, lb, PairingConstant::Bjerrum).0
                - bidh(&rm, &z, &d, &g, &f, lb, PairingConstant::Bjerrum).0)
                / (2.0 * h);
            assert_relative_eq!(a.eps, fd, max_relative = 1e-6);
        }
    }
}
