# Automatic differentiation

Automatic differentiation (AD) is used all throughout FeOs to calculate Helmholtz energy derivatives and Jacobians/Hessians for numerical solvers.
This section refers specifically to automatic (implicit) differentiation of phase equilibria with respect to model parameters that are crucial for parameter prediction or estimation.

Within Rust most phase equilibrium calculations can be used in an AD context, e.g., in order to calculate derivatives of process models (see [here](https://github.com/feos-org/feos-campd)). The Python interface focuses on the important use case of massively parallel phase equilibrium calculations for parameter estimation or prediction.

## Available equations of state
Only a subset of the models in FeOs can be used to calculate derivatives with respect to model parameters. However, those dedicated implementations also result in unprecedented performance (see example below). Similar to the [`EquationOfState`](eos.md#the-equationofstate-class) class, models with AD capabilities are collected in the `EquationOfStateAD` class. 

The currently available models are:

|Model|Description|Parameters pure|Parameters binary|
|-|-|-|-|
|`PcSaftNonAssoc`|The PC-SAFT equation of state including a dipolar contribution but no association|`m`, `sigma`, `epsilon_k`, `mu`|`k_ij`|
|`PcSaftFull`|The PC-SAFT equation of state with a dipolar contribution and association|`m`, `sigma`, `epsilon_k`, `mu`, `kappa_ab`, `epsilon_k_ab`, `na`, `nb`|`k_ij`|

## Properties
All properties that have parallel automatic differentiation with respect to model parameters enabled are available from the `Property` class

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    :toctree: generated/
    
    Property
```

Currently the following phase equilibrium properties are available in the AD interface of FeOs. We plan to extend the list in the future.

|Property|Pure/Binary|Inputs|
|-|-|-|
|`vapor_pressure`|Pure|Temperature|
|`boiling_temperature`|Pure|Pressure|
|`liquid_density`|Pure|Temperature, Pressure|
|`equilibrium_liquid_density`|Pure|Temperature|
|`enthalpy_of_vaporization`|Pure|Temperature|
|`residual_isobaric_heat_capacity`|Pure|Temperature, Pressure|
|`bubble_point_pressure`|Binary|Temperature, x1, Presssure estimate|
|`dew_point_pressure`|Binary|Temperature, y1, Presssure estimate|

For all properties `xyz`, `Property` contains a method `Property.xyz(eos, input)` which uses a model from the [`EquationOfState`](eos.md#the-equationofstate-class) class. This evaluation can be useful for comparisons of the same data to established models. For parameter estimation or learning, the `Property.xyz_derivatives(model, parameter_names, parameters, input)` methods can be used to determine values and derivatives with respect to the model parameters indicated in `parameter_names`.

## Electrolyte solutions (ePCP-SAFT)
`Property.electrolyte_properties_derivatives(parameters, derivatives, input, solvent=0, ...)` evaluates ePCP-SAFT for
every row of `input` (temperature in K, pressure in Pa, amounts of all components in mol; electroneutral) and returns
dictionaries of values and derivatives with respect to the requested parameters, and the convergence status:

|key|value|
|-|-|
|`ln_solvent_activity`|$\ln a_s$|
|`ln_gamma`|$\ln\gamma_i$, mole-fraction scale; solutes: infinite dilution in the pure solvent, solvent: pure solvent|
|`volume_difference`|volume of the solution minus volume of the pure solvent (m³)|
|`residual_heat_capacity_difference`|residual isobaric heat capacity of the solution minus that of the pure solvent (J/K)|
|`residual_enthalpy_difference`|residual enthalpy of the solution minus that of the pure solvent (J)|
|`relative_enthalpy`|enthalpy of the solution minus that of the same amounts at the reference states (solutes at infinite dilution, solvent pure), incl. the reaction enthalpies (J)|
|`moles`|amounts of all components, in chemical equilibrium if `reactions` are given (mol)|

`reactions` (optional) is a list of reaction equilibria, each as (stoichiometric coefficients of all components,
$\ln K_0$, $T_0$ in K, $\Delta H^\circ_0/R$ in K, $\Delta c^\circ_{p,0}/R$, $(\mathrm{d}\Delta c^\circ_p/\mathrm{d}T)/R$ in
1/K) with the equilibrium constant in the molality scale (`solvent_molar_mass` in kg/mol, default water) and
$\ln K(T)$ of Knopf et al. (2003, eq. 16); the amounts of a row are then the stoichiometric amounts. With reactions,
`residual_heat_capacity_difference` is the equilibrium heat capacity (central difference in temperature, see
`feos_core::ad::ElectrolyteProperties`).

Parameters are given as `(name, indices)`:

|name|indices|
|-|-|
|`m`, `sigma`, `epsilon_k`, `mu`, `q`|`[i]`|
|`dipole_scaling`, `polarizability_scaling`, `correlation_integral_parameter`|`[i]`|
|`k_ij`|`[i, j]`|
|`ion_pairing`|`[i, j]` (cation and anion; variants `"bidh"`, `"bimsa"`)|
|`born_delta_cation`, `born_delta_anion`, `dh_delta`|`[]`|

All options of `EquationOfState.epcpsaft` can be passed, including `epcpsaft_variant` (`"dh"`: Debye-Hückel,
`"msa"`: mean spherical approximation, `"bidh"`: Debye-Hückel with ion pairing, `"bimsa"`: binding mean spherical
approximation) and `msa_packing` (`"ions"`: packing term of the MSA/BiMSA over the ions only, `"total"`: over all
species). Ion pairing (BiDH, BiMSA) of a cation i and an anion j uses the association constant of
`ion_pairing_constant` (`"bjerrum"`, default; `"ebeling"`, analytic in l_B |z_i z_j| / d_ij; or `"shell"`, the
Bjerrum integral from d_ij to `ion_pairing_shell` d_ij, default 1.034) multiplied by exp(`ion_pairing`) of the binary
record of i and j (default 0). `bidh_dipole_factor` (default 0) includes the interaction of the dipole of the ion pairs
with the ionic atmosphere in BiDH (0.25 in Naseri Boroujeni et al. 2023); `bimsa_closure` selects the closure of BiMSA
(`"variational"`, default, or `"bernard_blum"`, the equations of Bernard and Blum 1996 with the pairs in the asymmetry
parameter). Rows are evaluated in parallel.

## Examples
The following example calculates pure-component vapor pressures including their derivatives with respect to the core PC-SAFT parameters for 10 Million temperatures in merely two seconds.

```python
import feos
import numpy as np

n = 10_000_000
fit_params = ["m", "sigma", "epsilon_k"]

# order: m, sigma, epsilon_k, mu
parameters = np.array([[1.5, 3.4, 230.0, 2.3]] * n)
temperature = np.expand_dims(np.linspace(250.0, 400.0, n), 1)
eos = feos.EquationOfStateAD.PcSaftNonAssoc
%time feos.Property.vapor_pressure_derivatives(eos, fit_params, parameters, temperature)
```
```
CPU times: user 1min 38s, sys: 611 ms, total: 1min 39s
Wall time: 1.98 s
```

For the most complex case, a binary mixture of cross-associating mixtures, the following example calculates 100.000 bubble point pressures and their derivative with respect to the binary interaction parameter in less than 5 seconds.
```python
import feos
import numpy as np

n = 100_000
fit_params = ["k_ij"]
parameters = np.array([[
    # Substance 1: m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
    1.5, 3.4, 230.0, 2.3, 0.01, 1200.0, 1.0, 2.0, 
    # Substance 2: m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
    2.3, 3.5, 245.0, 1.4, 0.005, 500.0, 1.0, 1.0,
    # k_ij
    0.01,
]] * n)
temperature = np.linspace(200.0, 388.0, n)
molefracs = np.array([0.5] * n)
pressure = np.array([1e5] * n)
input = np.stack((temperature, molefracs, pressure), axis=1)
eos = feos.EquationOfStateAD.PcSaftFull
%time feos.Property.bubble_point_pressure_derivatives(eos, fit_params, parameters, input)
```
```
CPU times: user 4min 52s, sys: 87.1 ms, total: 4min 53s
Wall time: 4.74 s
```