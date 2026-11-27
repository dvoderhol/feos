use crate::geometry::Grid;
use feos_core::{FeosError, FeosResult, ReferenceSystem};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Dimension, Zip};
use num_dual::DualNum;
use quantity::{Energy, Temperature};

/// Mean-field interaction between the molecules in disjoint regions of the grid,
///
/// $$F^\mathrm{mf}=\frac{1}{2}\sum_{kl}W_{kl}N_kN_l,\qquad N_k=\sum_i\int_{V_k}\rho_i(\mathbf{r})\mathrm{d}\mathbf{r}$$
///
/// e.g. interactions between molecules at different adsorption sites that are not described by
/// the Helmholtz energy functional. A molecule in region $k$ feels the potential
/// $v_k=\sum_lW_{kl}N_l$, which enters the Euler-Lagrange equation in addition to the external
/// potential. The term is included in the grand potential and, like the external potential, it
/// does not depend on temperature. Only for molecules with one segment ($m=1$).
#[derive(Clone)]
pub struct RegionMeanField<D: Dimension> {
    /// Region of every grid point; negative values: the point belongs to no region.
    pub regions: Array<isize, D>,
    /// Symmetric interaction energies $W_{kl}$ between molecules in regions $k$ and $l$.
    pub coupling: Energy<Array2<f64>>,
}

impl<D: Dimension> RegionMeanField<D> {
    pub fn new(regions: Array<isize, D>, coupling: Energy<Array2<f64>>) -> FeosResult<Self> {
        let w = coupling.to_reduced();
        let n = w.nrows();
        if w.ncols() != n {
            return Err(FeosError::Error(format!(
                "RegionMeanField: the coupling matrix must be square (shape {:?}).",
                w.shape()
            )));
        }
        if regions.iter().any(|&k| k >= n as isize) {
            return Err(FeosError::Error(format!(
                "RegionMeanField: region indices must be smaller than the size of the coupling matrix ({n})."
            )));
        }
        let scale = w.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if Zip::from(&w)
            .and(&w.t())
            .fold(false, |asym, &a, &b| asym || (a - b).abs() > 1e-12 * scale)
        {
            return Err(FeosError::Error(
                "RegionMeanField: the coupling matrix must be symmetric.".into(),
            ));
        }
        Ok(Self { regions, coupling })
    }

    /// Reduced form for a profile on `grid` at `temperature`.
    pub(crate) fn reduce(
        &self,
        grid: &Grid,
        temperature: Temperature,
    ) -> FeosResult<ReducedRegionMeanField<D>> {
        let shape: Vec<_> = grid.axes().iter().map(|ax| ax.grid.len()).collect();
        if self.regions.shape() != shape.as_slice() {
            return Err(FeosError::Error(format!(
                "RegionMeanField: the regions have shape {:?}, the grid has {:?} points.",
                self.regions.shape(),
                shape
            )));
        }
        let (axis_weights, functional_determinant) = grid.integration_weights();
        let mut weights = Array::from_elem(self.regions.raw_dim(), functional_determinant);
        for (i, w) in axis_weights.into_iter().enumerate() {
            for mut l in weights.lanes_mut(Axis(i)) {
                l *= w;
            }
        }
        Ok(ReducedRegionMeanField {
            regions: self.regions.clone(),
            coupling: self.coupling.to_reduced() / temperature.to_reduced(),
            weights,
        })
    }
}

/// [RegionMeanField] with the coupling divided by $k_\mathrm{B}T$ of the profile and the
/// integration weights of the grid.
#[derive(Clone)]
pub(crate) struct ReducedRegionMeanField<D: Dimension> {
    regions: Array<isize, D>,
    coupling: Array2<f64>,
    weights: Array<f64, D>,
}

impl<D: Dimension> ReducedRegionMeanField<D> {
    pub(crate) fn into_dyn(self) -> ReducedRegionMeanField<ndarray::IxDyn> {
        ReducedRegionMeanField {
            regions: self.regions.into_dyn(),
            coupling: self.coupling,
            weights: self.weights.into_dyn(),
        }
    }

    /// Number of molecules $N_k$ in every region for the total density `density` (sum over all
    /// segments).
    pub(crate) fn moles<N: DualNum<Primitive = f64> + Copy>(
        &self,
        density: ArrayView<N, D>,
    ) -> Array1<N> {
        let mut n = Array1::from_elem(self.coupling.nrows(), N::zero());
        Zip::from(&density)
            .and(&self.weights)
            .and(&self.regions)
            .for_each(|&rho, &w, &k| {
                if k >= 0 {
                    n[k as usize] += rho * w;
                }
            });
        n
    }

    /// Potential $\beta v_k$ (at the temperature of the profile, multiplied by `ratio` like the
    /// external potential) on every grid point for the total density `density`.
    pub(crate) fn potential<N: DualNum<Primitive = f64> + Copy>(
        &self,
        density: ArrayView<N, D>,
        ratio: N,
    ) -> Array<N, D> {
        let n = self.moles(density);
        let v: Vec<N> = self
            .coupling
            .outer_iter()
            .map(|w| w.iter().zip(&n).fold(N::zero(), |acc, (&w, &n)| acc + n * w) * ratio)
            .collect();
        self.regions
            .mapv(|k| if k >= 0 { v[k as usize] } else { N::zero() })
    }
}
