//! Association contribution with model parameters of the generic type `D`.
//!
//! Same equations and solution procedure as `crate::association` (analytic solutions for one
//! site type, Newton iteration of Michelsen (2006) for cross association); the association
//! strength is evaluated with the parameters in `EPcpSaftPars<D>`.
use crate::epcpsaft::parameters::{EPcpSaftAssociationRecord, EPcpSaftPars};
use feos_core::parameter::{AssociationParameters, AssociationSite, BinaryParameters};
use feos_core::{FeosError, FeosResult, StateHD};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_dual::linalg::LU;

#[derive(Clone, Copy)]
pub struct Association {
    max_iter: usize,
    tol: f64,
}

impl Association {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Association strength matrix: g_HS(d_ij) (exp(eps_AB/kT) - 1) kappa_AB (sigma_i sigma_j)^1.5
    fn association_strength<D: DualNum<f64> + Copy>(
        p: &EPcpSaftPars<D>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        (sites1, sites2): (&[AssociationSite], &[AssociationSite]),
        association_parameters: &[BinaryParameters<EPcpSaftAssociationRecord, ()>],
    ) -> DMatrix<D> {
        let mut delta = DMatrix::zeros(sites1.len(), sites2.len());
        if sites1.len() * sites2.len() == 0 {
            return delta;
        }

        let [zeta2, n3] = p.packing(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        for b in association_parameters {
            let [i, j] = [b.id1, b.id2];
            let [comp_i, comp_j] = [sites1[i].assoc_comp, sites2[j].assoc_comp];
            // g_HS(d)
            let di = diameter[comp_i];
            let dj = diameter[comp_j];
            let k = di * dj / (di + dj) * (n2 * n3i);
            let g_contact = n3i * (k * (k / 18.0 + 0.5) + 1.0);

            let f_ab = (state.temperature.recip() * b.model_record.epsilon_k_ab).exp_m1();
            let k_ab = (p.sigma[comp_i] * p.sigma[comp_j]).powf(1.5) * b.model_record.kappa_ab;
            delta[(i, j)] = g_contact * (f_ab * k_ab);
        }
        delta
    }

    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        p: &EPcpSaftPars<D>,
        a: &AssociationParameters<EPcpSaftAssociationRecord>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> D {
        let delta_ab =
            Self::association_strength(p, state, diameter, (&a.sites_a, &a.sites_b), &a.binary_ab);
        let delta_cc =
            Self::association_strength(p, state, diameter, (&a.sites_c, &a.sites_c), &a.binary_cc);

        match (a.sites_a.len() * a.sites_b.len(), a.sites_c.len()) {
            (0, 0) => D::zero(),
            (1, 0) => Self::helmholtz_energy_density_ab_analytic(a, state, delta_ab[(0, 0)]),
            (0, 1) => Self::helmholtz_energy_density_cc_analytic(a, state, delta_cc[(0, 0)]),
            (1, 1) => {
                Self::helmholtz_energy_density_ab_analytic(a, state, delta_ab[(0, 0)])
                    + Self::helmholtz_energy_density_cc_analytic(a, state, delta_cc[(0, 0)])
            }
            _ => {
                // site densities of associating segments
                let rho: Vec<_> = a
                    .sites_a
                    .iter()
                    .chain(a.sites_b.iter())
                    .chain(a.sites_c.iter())
                    .map(|s| state.partial_density[a.component_index[s.assoc_comp]] * s.n)
                    .collect();
                let rho = DVector::from(rho);
                self.helmholtz_energy_density_cross_association(&rho, &delta_ab, &delta_cc)
                    .unwrap_or_else(|_| D::from(f64::NAN))
            }
        }
    }

    fn helmholtz_energy_density_ab_analytic<A, D: DualNum<f64> + Copy>(
        a: &AssociationParameters<A>,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        // site densities
        let rhoa =
            state.partial_density[a.component_index[a.sites_a[0].assoc_comp]] * a.sites_a[0].n;
        let rhob =
            state.partial_density[a.component_index[a.sites_b[0].assoc_comp]] * a.sites_b[0].n;

        // fraction of non-bonded association sites
        let sqrt = ((delta * (rhoa - rhob) + 1.0).powi(2) + delta * rhob * 4.0).sqrt();
        let (xa, xb) = if rhoa > rhob {
            let xb = (sqrt + delta * (rhoa - rhob) + 1.0).recip() * 2.0;
            let xa = (delta * rhob * xb + 1.0).recip();
            (xa, xb)
        } else {
            let xa = (sqrt + delta * (rhob - rhoa) + 1.0).recip() * 2.0;
            let xb = (delta * rhoa * xa + 1.0).recip();
            (xa, xb)
        };

        rhoa * (xa.ln() - xa * 0.5 + 0.5) + rhob * (xb.ln() - xb * 0.5 + 0.5)
    }

    fn helmholtz_energy_density_cc_analytic<A, D: DualNum<f64> + Copy>(
        a: &AssociationParameters<A>,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        // site density
        let rhoc =
            state.partial_density[a.component_index[a.sites_c[0].assoc_comp]] * a.sites_c[0].n;

        // fraction of non-bonded association sites
        let xc = ((delta * 4.0 * rhoc + 1.0).sqrt() + 1.0).recip() * 2.0;

        rhoc * (xc.ln() - xc * 0.5 + 0.5)
    }

    fn helmholtz_energy_density_cross_association<D: DualNum<f64> + Copy>(
        &self,
        rho: &DVector<D>,
        delta_ab: &DMatrix<D>,
        delta_cc: &DMatrix<D>,
    ) -> FeosResult<D> {
        // check if density is close to 0
        if rho.sum().re() < f64::EPSILON {
            return Ok(D::zero());
        }

        // cross-association according to Michelsen2006
        let mut x = DVector::from_element(rho.len(), 0.2);
        let delta_ab_re = delta_ab.map(|d| d.re());
        let delta_cc_re = delta_cc.map(|d| d.re());
        let rho_re = rho.map(|r| r.re());
        for k in 0..self.max_iter {
            if newton_step_cross_association(&mut x, &delta_ab_re, &delta_cc_re, &rho_re, self.tol)?
            {
                break;
            }
            if k == self.max_iter - 1 {
                return Err(FeosError::NotConverged("Cross association".into()));
            }
        }

        // derivatives
        let mut x_dual = x.map(D::from);
        for _ in 0..D::NDERIV {
            newton_step_cross_association(&mut x_dual, delta_ab, delta_cc, rho, self.tol)?;
        }

        let f = |x: D| x.ln() - x * 0.5 + 0.5;
        Ok(rho.dot(&x_dual.map(f)))
    }
}

fn newton_step_cross_association<D: DualNum<f64> + Copy>(
    x: &mut DVector<D>,
    delta_ab: &DMatrix<D>,
    delta_cc: &DMatrix<D>,
    rho: &DVector<D>,
    tol: f64,
) -> FeosResult<bool> {
    let nassoc = x.len();
    // gradient
    let mut g = x.map(|x| x.recip());
    // Hessian
    let mut h: DMatrix<D> = DMatrix::zeros(nassoc, nassoc);

    // split arrays
    let (a, b) = delta_ab.shape();
    let (c, _) = delta_cc.shape();
    let (xa, xc) = x.rows_range_pair(..a + b, a + b..);
    let (xa, xb) = xa.rows_range_pair(..a, a..);
    let (rhoa, rhoc) = rho.rows_range_pair(..a + b, a + b..);
    let (rhoa, rhob) = rhoa.rows_range_pair(..a, a..);

    for i in 0..nassoc {
        // gradients
        let dnx = if i < a {
            let d = delta_ab.row(i).transpose();
            xb.component_mul(&rhob).dot(&d) + 1.0
        } else if i < a + b {
            let d = delta_ab.column(i - a);
            xa.component_mul(&rhoa).dot(&d) + 1.0
        } else {
            let d = delta_cc.column(i - a - b);
            xc.component_mul(&rhoc).dot(&d) + 1.0
        };
        g[i] -= dnx;

        // approximate hessian
        h[(i, i)] = -dnx / x[i];
        if i < a {
            for j in 0..b {
                h[(i, a + j)] = -delta_ab[(i, j)] * rhob[j];
            }
        } else if i < a + b {
            for j in 0..a {
                h[(i, j)] = -delta_ab[(j, i - a)] * rhoa[j];
            }
        } else {
            for j in 0..c {
                h[(i, a + b + j)] -= delta_cc[(i - a - b, j)] * rhoc[j];
            }
        }
    }

    // Newton step; avoid stepping to negative values for x (see Michelsen 2006)
    let delta_x = LU::new(h)?.solve(&g);
    x.iter_mut().zip(&delta_x).for_each(|(x, &delta_x)| {
        if delta_x.re() < x.re() * 0.8 {
            *x -= delta_x
        } else {
            *x *= 0.2
        }
    });

    // convergence
    Ok(g.map(|g| g.re()).norm() < tol)
}
