#![cfg(feature = "dft")]
#![cfg(feature = "pcsaft")]
use approx::assert_relative_eq;
use feos::epcpsaft::{EPcpSaftFunctional, EPcpSaftParameters};
use feos::pcsaft::{PcSaftFunctional, PcSaftParameters};
use feos_core::PhaseEquilibrium;
use feos_core::parameter::IdentifierOption;
use feos_dft::interface::Interface;
use quantity::*;
use std::error::Error;

/// Without ions, the ePCP-SAFT functional equals the PC-SAFT functional (bulk phase equilibrium
/// and planar interface), for a non-associating and an associating component.
#[test]
fn test_dft_neutral_equals_pcsaft() -> Result<(), Box<dyn Error>> {
    for (component, temperature) in [("propane", 200.0 * KELVIN), ("water_np", 350.0 * KELVIN)] {
        let path = "tests/pcsaft/test_parameters.json";
        let pcsaft = PcSaftFunctional::new(PcSaftParameters::from_json(
            vec![component],
            path,
            None,
            IdentifierOption::Name,
        )?);
        let epcpsaft = EPcpSaftFunctional::new(EPcpSaftParameters::from_json(
            vec![component],
            path,
            None,
            IdentifierOption::Name,
        )?)?;
        let vle_pcsaft = PhaseEquilibrium::pure(&&pcsaft, temperature, None, Default::default())?;
        let vle_epcpsaft =
            PhaseEquilibrium::pure(&&epcpsaft, temperature, None, Default::default())?;
        assert_relative_eq!(
            vle_epcpsaft.liquid().density,
            vle_pcsaft.liquid().density,
            max_relative = 1e-13
        );
        let w = 100.0 * ANGSTROM;
        let t_init = 500.0 * KELVIN;
        let st_pcsaft = Interface::planar_from_tanh(&vle_pcsaft, 512, w, t_init)
            .solve(None)?
            .surface_tension
            .unwrap();
        let st_epcpsaft = Interface::planar_from_tanh(&vle_epcpsaft, 512, w, t_init)
            .solve(None)?
            .surface_tension
            .unwrap();
        assert_relative_eq!(st_epcpsaft, st_pcsaft, max_relative = 1e-10);
    }
    Ok(())
}
