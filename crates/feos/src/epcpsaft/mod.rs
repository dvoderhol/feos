//! ePCP-SAFT: PCP-SAFT with ion-ion contributions (Debye-Hückel, mean spherical approximation, or
//! either with ion pairing: BiDH, BiMSA), a Born contribution and the permittivity model of
//! Rueben et al. (2024)
//!
//! [Gross et al. (2001)](https://doi.org/10.1021/ie0003887)

mod eos;
pub(crate) mod parameters;

pub use eos::{
    BornDiameter, DQVariants, ELECTROLYTE_GRADIENT_CHUNK, EPcpSaft, EPcpSaftAD, EPcpSaftOptions,
    EPcpSaftParameter, EPcpSaftVariants, ElectrolyteDerivatives, MsaPacking, PairingConstant,
    ValueGradient, electrolyte_properties_derivatives,
};
pub use parameters::EPcpSaftPars;
pub use parameters::{
    EPcpSaftAssociationRecord, EPcpSaftBinaryRecord, EPcpSaftParameters, EPcpSaftRecord,
    PermittivityRecord,
};
