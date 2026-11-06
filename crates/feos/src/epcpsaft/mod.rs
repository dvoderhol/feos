//! ePCP-SAFT: PCP-SAFT with Debye-Hückel and Born contributions and the permittivity
//! model of Rueben et al. (2024)
//!
//! [Gross et al. (2001)](https://doi.org/10.1021/ie0003887)

#[cfg(feature = "dft")]
mod dft;
mod eos;
pub(crate) mod parameters;

#[cfg(feature = "dft")]
pub use dft::{EPcpSaftFunctional, EPcpSaftFunctionalContribution};
pub use eos::{
    DQVariants, ELECTROLYTE_GRADIENT_CHUNK, EPcpSaft, EPcpSaftAD, EPcpSaftBinary, EPcpSaftOptions,
    EPcpSaftParameter, EPcpSaftPure, EPcpSaftVariants, ElectrolyteDerivatives, MsaPacking,
    ValueGradient, electrolyte_properties_derivatives,
};
pub use parameters::EPcpSaftPars;
pub use parameters::{
    EPcpSaftAssociationRecord, EPcpSaftBinaryRecord, EPcpSaftParameters, EPcpSaftRecord,
    PermittivityRecord,
};
