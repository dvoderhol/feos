use crate::epcpsaft::parameters::EPcpSaftPars;
use feos_core::StateHD;
use num_dual::DualNum;
use quantity::{COULOMB, EPSILON0, FARAD, JOULE, KB, KELVIN, METER, QE};
use std::f64::consts::PI;

/// Vacuum permittivity in F/m (`quantity::EPSILON0`).
pub fn epsilon_0() -> f64 {
    (EPSILON0 / (FARAD / METER)).into_value()
}

impl<D: DualNum<Primitive = f64> + Copy> EPcpSaftPars<D> {
    /// Relative permittivity of the mixture (Rueben et al. 2024, perturbation theory of Neumaier et al. 2022).
    ///
    /// eps_r = 1 + 3y [1 + y + (17/16 I - 1) y^2],  I = 1 + a2 (exp(-y) - 1),
    /// y = 4 pi / 9 sum_i rho_i [(a11 mu^2)_i / (kB T) + 3 (a12 alpha)_i],  a2 = sum_i x_i a2_i,
    ///
    /// plus the offset sum_k c_k tau^k, tau = (T/K - 298.15)/100, with the coefficients `permittivity_offset`
    /// (zero by default), multiplied by rho_c / rho_ref(T) if `permittivity_offset_density` is not zero
    /// ([Self::permittivity_offset_scale]).
    ///
    /// Returns `None` if at least one component has no permittivity record.
    pub fn permittivity(&self, state: &StateHD<D>) -> Option<D> {
        let [
            dipole_scaling,
            polarizability_scaling,
            correlation_integral_parameter,
        ] = self.permittivity.as_ref()?;
        // 1 D^2 = 1e-19 J A^3, as in the dipole term of PCP-SAFT
        let mu2_factor = 1e-19 * (JOULE / KELVIN / KB).into_value();
        let t_inv = state.temperature.recip();

        let mut y = D::zero();
        let mut a2 = D::zero();
        for i in 0..dipole_scaling.len() {
            y += state.partial_density[i]
                * (t_inv * dipole_scaling[i] * mu2_factor + polarizability_scaling[i] * 3.0);
            a2 += state.molefracs[i] * correlation_integral_parameter[i];
        }
        y *= 4.0 / 9.0 * PI;

        let correlation_integral = ((-y).exp() - 1.0) * a2 + 1.0;
        let eps = y * 3.0 * (y * y * (correlation_integral * (17.0 / 16.0) - 1.0) + y + 1.0) + 1.0;
        Some(
            eps + self.permittivity_offset_value(state.temperature)
                * self.permittivity_offset_scale(state),
        )
    }

    /// Permittivity offset sum_k c_k tau^k, tau = (T/K - 298.15)/100 (Horner scheme).
    pub fn permittivity_offset_value(&self, temperature: D) -> D {
        let tau = (temperature - 298.15) / 100.0;
        self.permittivity_offset
            .iter()
            .rev()
            .fold(D::zero(), |acc, &c| acc * tau + c)
    }

    /// Factor of the permittivity offset: rho_c / rho_ref(T) with the density rho_c of the component
    /// `permittivity_offset_component` and rho_ref(T) = sum_k d_k tau^k (coefficients `permittivity_offset_density`,
    /// 1/A^3; the offset vanishes at zero density and without that component); 1 if all d_k are zero.
    pub fn permittivity_offset_scale(&self, state: &StateHD<D>) -> D {
        if self.permittivity_offset_density.iter().all(|&d| d == 0.0) {
            return D::one();
        }
        let tau = (state.temperature - 298.15) / 100.0;
        let rho_ref = self
            .permittivity_offset_density
            .iter()
            .rev()
            .fold(D::zero(), |acc, &d| acc * tau + d);
        state.partial_density[self.permittivity_offset_component] / rho_ref
    }

    /// Bjerrum length e^2 / (4 pi eps_0 eps_r kB T) in units of Angstrom.
    pub fn bjerrum_length(&self, temperature: D, permittivity: D) -> D {
        let qe = (QE / COULOMB).into_value();
        let kb = (KB / (JOULE / KELVIN)).into_value();
        (permittivity * temperature * 4.0 * PI * epsilon_0() * kb).recip() * qe * qe * 1e10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcpsaft::parameters::utils::water_nacl_parameters;
    use approx::assert_relative_eq;
    use nalgebra::dvector;
    use num_dual::Dual2_64;

    #[test]
    fn vacuum_permittivity() {
        assert_relative_eq!(epsilon_0(), 8.8541878188e-12, max_relative = 1e-15);
    }

    #[test]
    fn permittivity_formula() {
        // reference: explicit evaluation of the formula in the docstring
        let p = EPcpSaftPars::new(&water_nacl_parameters()).unwrap();
        let t = 298.15;
        let v = 31.875;
        let s = StateHD::new(t, v, &dvector![0.9, 0.05, 0.05]);
        let eps = p.permittivity(&s).unwrap();

        let rho = [0.9 / v, 0.05 / v, 0.05 / v];
        let [a, b, c] = p.permittivity.as_ref().unwrap();
        let kb = 1.380649e-23;
        let y: f64 = (0..3)
            .map(|i| rho[i] * (a[i] * 1e-19 / (kb * t) + 3.0 * b[i]))
            .sum::<f64>()
            * 4.0
            * PI
            / 9.0;
        let a2: f64 = (0..3).map(|i| rho[i] * v * c[i]).sum();
        let ci = 1.0 + a2 * ((-y).exp() - 1.0);
        let eps_ref = 1.0 + 3.0 * y * (1.0 + y + (17.0 / 16.0 * ci - 1.0) * y * y);
        assert_relative_eq!(eps, eps_ref, max_relative = 1e-14);
    }

    #[test]
    fn permittivity_offset() {
        // the offset adds sum_k c_k tau^k with its temperature derivatives
        let mut p = EPcpSaftPars::new(&water_nacl_parameters()).unwrap();
        let (t, v, x) = (318.15, 31.875, dvector![0.9, 0.05, 0.05]);
        let without = |t: Dual2_64| {
            let s = StateHD::new(t, Dual2_64::from(v), &x.map(Dual2_64::from));
            p.map(Dual2_64::from).permittivity(&s).unwrap()
        };
        let e0 = without(Dual2_64::new(t, 1.0, 0.0));
        let c = [
            0.4, -1.3, 0.25, 0.05, -0.02, 0.0, 0.0, 0.001, -3e-4, 0.0, 2e-5, -1e-6,
        ];
        p.permittivity_offset = c;
        let s = StateHD::new(
            Dual2_64::new(t, 1.0, 0.0),
            Dual2_64::from(v),
            &x.map(Dual2_64::from),
        );
        let e1 = p.map(Dual2_64::from).permittivity(&s).unwrap();
        let tau = (t - 298.15) / 100.0;
        let value: f64 = (0..12).map(|k| c[k] * tau.powi(k as i32)).sum();
        let first: f64 = (1..12)
            .map(|k| c[k] * k as f64 * tau.powi(k as i32 - 1))
            .sum::<f64>()
            / 100.0;
        let second: f64 = (2..12)
            .map(|k| c[k] * (k * (k - 1)) as f64 * tau.powi(k as i32 - 2))
            .sum::<f64>()
            / 1e4;
        assert_relative_eq!(e1.re - e0.re, value, max_relative = 1e-12);
        assert_relative_eq!(e1.v1 - e0.v1, first, max_relative = 1e-10);
        assert_relative_eq!(e1.v2 - e0.v2, second, max_relative = 1e-8);
    }

    #[test]
    fn permittivity_offset_density_scale() {
        // offset c(T) rho_0 / rho_ref(T): value, zero without the component, derivatives with respect to T and V
        let mut p = EPcpSaftPars::new(&water_nacl_parameters()).unwrap();
        let c = [0.4, -1.3, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = [
            0.0345, -0.0012, -0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let eps = |p: &EPcpSaftPars<f64>, t: f64, v: f64, x: &nalgebra::DVector<f64>| {
            p.permittivity(&StateHD::new(t, v, x)).unwrap()
        };
        let (t, v, x) = (318.15, 31.875, dvector![0.9, 0.05, 0.05]);
        let e0 = eps(&p, t, v, &x);
        p.permittivity_offset = c;
        p.permittivity_offset_density = d;
        let tau = (t - 298.15) / 100.0;
        let poly = |a: &[f64; 12]| (0..12).map(|k| a[k] * tau.powi(k as i32)).sum::<f64>();
        let rho_0 = x[0] / v;
        assert_relative_eq!(
            eps(&p, t, v, &x) - e0,
            poly(&c) * rho_0 / poly(&d),
            max_relative = 1e-12
        );
        // without water the offset vanishes
        let ions = dvector![0.0, 0.5, 0.5];
        let mut q = p.clone();
        q.permittivity_offset = [0.0; 12];
        assert_relative_eq!(
            eps(&p, t, v, &ions),
            eps(&q, t, v, &ions),
            max_relative = 1e-15
        );
        // derivatives of the dual-number evaluation against central differences
        let pd = p.map(Dual2_64::from);
        let dt = pd
            .permittivity(&StateHD::new(
                Dual2_64::new(t, 1.0, 0.0),
                Dual2_64::from(v),
                &x.map(Dual2_64::from),
            ))
            .unwrap();
        let dv = pd
            .permittivity(&StateHD::new(
                Dual2_64::from(t),
                Dual2_64::new(v, 1.0, 0.0),
                &x.map(Dual2_64::from),
            ))
            .unwrap();
        let h = 1e-4;
        let fd_t = (eps(&p, t + h, v, &x) - eps(&p, t - h, v, &x)) / (2.0 * h);
        let fd_v = (eps(&p, t, v + h, &x) - eps(&p, t, v - h, &x)) / (2.0 * h);
        assert_relative_eq!(dt.v1, fd_t, max_relative = 1e-7);
        assert_relative_eq!(dv.v1, fd_v, max_relative = 1e-7);
    }
}
