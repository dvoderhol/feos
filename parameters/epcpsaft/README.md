# ePCP-SAFT Parameters

This directory contains files with parameters for the ePCP-SAFT equation of state: PCP-SAFT with the Debye-Hückel and Born contributions of ePC-SAFT and the permittivity model of [Rueben et al. (2024b)](https://doi.org/10.1021/acs.jced.4c00347).
The publications are provided in the [`literature.bib`](literature.bib) file.

All PCP-SAFT parameters are identical to the files in [`../pcsaft`](../pcsaft). Only parameters that do not exist in PCP-SAFT are added.

## Pure Substance Parameters

| file | description | publication(s) |
| - | - | :-: |
| [`esper2023.json`](esper2023.json) | copy of [`../pcsaft/esper2023.json`](../pcsaft/esper2023.json); permittivity parameters added for 747 substances | [&#128279;](https://doi.org/10.1021/acs.iecr.3c02255) |

## Binary Parameters

| file | description | publication(s) |
| - | - | :-: |
| [`rehner2023_binary.json`](rehner2023_binary.json) | identical copy of [`../pcsaft/rehner2023_binary.json`](../pcsaft/rehner2023_binary.json) | [&#128279;](https://doi.org/10.1007/s10765-023-03290-3) |

## Permittivity Parameters

`permittivity_record` with `PerturbationTheory` parameters ([Rueben et al. 2022](https://doi.org/10.1016/j.fluid.2021.113346), [Rueben et al. 2024a](https://doi.org/10.1021/acs.jced.3c00323)) for 747 substances,
fitted with the liquid density of the PCP-SAFT parameters of the respective substance by
[`examples/epcpsaft/fit_solvent_permittivity.py`](../../examples/epcpsaft/fit_solvent_permittivity.py):
DDB data (all data points after the filter of Rueben et al. 2024a), Huber loss, scheme selection of Rueben et al. (2024a);
water: scheme P prescribed (`dipole_scaling` 5.194 D², `correlation_integral_parameter` 0.1311).

## Ions

No ion parameters yet. All ion parameters (`m = 1`, `sigma`, `epsilon_k`, `z`, `permittivity_record` and ion-solvent `k_ij`) are fitted for ePCP-SAFT.
