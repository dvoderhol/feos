"""feos side: periodic 3D grid + MLIP V_ext -> PC-SAFT cDFT adsorption isotherms.

feos expects the external potential as an Energy array of shape (n_segments, n_a, n_b, n_c) on
`Grid.periodical_3d`, whose points are at fractional coordinates (i + 0.5)/n of the unit cell.
For PC-SAFT the first axis is the component; feos divides V_ext by the segment number m, so a
molecule-centred MLIP V_ext is used exactly only for m = 1 (e.g. CH4 with gross2001 parameters).
feos returns absolute adsorption (moles in the cell volume); per-mass values use the framework
mass of the unit cell.
"""
import numpy as np
import si_units as si
import feos

AMU_KG = 1.66053906660e-27


def periodic_grid(cellpar, n):
    a, b, c, al, be, ga = cellpar
    return feos.Grid.periodical_3d(list(n), [a * si.ANGSTROM, b * si.ANGSTROM, c * si.ANGSTROM],
                                   [al * si.DEGREES, be * si.DEGREES, ga * si.DEGREES])


def external_potential(vext_k):
    """V_ext / K (n_a, n_b, n_c) -> feos Energy array (1, n_a, n_b, n_c); inf is capped by feos at 50 kT."""
    v = np.where(np.isfinite(vext_k), vext_k, 1e8)
    return v[None] * si.KB * si.KELVIN


def pcsaft(components, parameter_file, fmt_version=None):
    """PC-SAFT functional. fmt_version: feos.FMTVersion (default WhiteBear). For strongly localised adsorbed
    molecules (deep, narrow wells) AntiSymWhiteBear has the correct zero-dimensional (single-cavity) limit and
    clamps |n2v| <= n2, which keeps FFT ringing of the weighted densities from breaking the solver."""
    params = feos.Parameters.from_json(components, parameter_file)
    if fmt_version is None:
        return feos.HelmholtzEnergyFunctional.pcsaft(params)
    return feos.HelmholtzEnergyFunctional.pcsaft(params, fmt_version=fmt_version)


def henry_from_vext(vext_k, cellpar, temperature, framework_mass_amu):
    """K_H in mol/(kg Pa) directly from V_ext: K_H = <exp(-V/kT)>_cell * V_cell / (m_framework R T)."""
    from ase.cell import Cell
    vol_m3 = Cell.fromcellpar(cellpar).volume * 1e-30
    with np.errstate(over="ignore"):
        boltz = np.mean(np.exp(-np.where(np.isfinite(vext_k), vext_k, np.inf) / temperature))
    return boltz * vol_m3 / (framework_mass_amu * AMU_KG * 8.31446262 * temperature)


class _Pore3DBase:
    """Common interface mirroring feos.Pore1D: `.grid`, `.external_potential`, `.initialize(bulk, ...)`.

    Use exactly like Pore1D in the feos examples, e.g.
        Adsorption.adsorption_isotherm(func, T, p, pore.grid, pore.external_potential)
    """

    def __init__(self, cellpar, vext_k, framework_mass_amu, temperature=None, m=1.0):
        self.m = m
        self.cellpar = np.asarray(cellpar, dtype=float)
        self.temperature = temperature
        self.framework_mass_amu = framework_mass_amu
        self.vext_k = vext_k
        self.grid = periodic_grid(self.cellpar, vext_k.shape)
        self.external_potential = external_potential(vext_k)

    def initialize(self, bulk, density=None, specification=None):
        if self.temperature is not None:
            t_bulk = float(bulk.temperature / si.KELVIN)
            if abs(t_bulk - self.temperature) > 1e-6:
                import warnings
                warnings.warn(f"V_ext was averaged at {self.temperature} K but the bulk state is at {t_bulk} K")
        spec = specification if specification is not None else feos.PoreSpecification.ChemicalPotential
        return feos.PoreProfile(self.grid, bulk, self.external_potential, density=density, specification=spec)

    def henry_coefficient(self, temperature_k):
        """K_H in mol/(kg Pa) from the grid potential (equals feos PoreProfile.henry_coefficients).

        Only defined for m = 1, like feos: for chain molecules the bond integrals smear the potential,
        so the isotherm slope is not <exp(-beta V)> (feos' henry_coefficients panics for m != 1)."""
        if self.m != 1.0:
            raise ValueError(f"Henry coefficient from V_ext is only defined for m = 1 (m = {self.m})")
        return henry_from_vext(self.vext_k, self.cellpar, temperature_k, self.framework_mass_amu)

    def per_mass(self, moles):
        """Convert feos (absolute) moles in the cell to mol/kg framework."""
        return np.asarray(moles / si.MOL) / (self.framework_mass_amu * AMU_KG)


class MLIPPore3D(_Pore3DBase):
    """Periodic 3D pore with V_ext from MLIP insertion energies (feos_mlip.vext.VextResult).

    The Boltzmann orientation average makes V_ext temperature dependent, so (unlike feos.Pore1D)
    the temperature is fixed at construction. The MLIP grid is interpolated onto the feos grid of
    spacing `spacing` (periodic cubic interpolation of beta*V, capped at `cap`).
    """

    def __init__(self, vext_result, temperature, spacing=None, n_grid=None, cap=50.0, framework_mass_amu=None, m=1.0):
        from .vext import interpolate_periodic
        v = vext_result.vext(temperature)
        if spacing is not None or n_grid is not None:
            n_grid = n_grid or [int(np.ceil(L / spacing)) for L in vext_result.cellpar[:3]]
            bv = interpolate_periodic(v / temperature, n_grid, cap=cap)
            v = np.where(bv >= cap, np.inf, bv * temperature)
        mass = framework_mass_amu or vext_result.info.get("framework_mass_amu")
        super().__init__(vext_result.cellpar, v, mass, temperature, m=m)
        self.vext_result = vext_result


class LJPore3D(_Pore3DBase):
    """Standard approach (feos' former Pore3D): UFF/LJ 12-6 framework potential, Lorentz-Berthelot,
    cutoff 14 A, times the segment number m. `atoms` is the framework unit cell (ASE Atoms)."""

    def __init__(self, atoms, sigma_f, epsilon_k_f, m=1.0, spacing=0.3, n_grid=None, cutoff=14.0, cap_k=None):
        from ase.cell import Cell
        from .lj import lj_external_potential
        from .vext import grid_points
        par = atoms.cell.cellpar()
        at = atoms.copy()
        at.set_cell(Cell.fromcellpar(par), scale_atoms=True)  # feos frame: a || x, b in the xy plane
        n_grid = n_grid or [int(np.ceil(L / spacing)) for L in par[:3]]
        v = lj_external_potential(grid_points(n_grid).reshape(-1, 3) @ at.cell.array, at, sigma_f, epsilon_k_f,
                                  m=m, cutoff=cutoff).reshape(n_grid)
        if cap_k is not None:
            v = np.where(v < cap_k, v, np.inf)
        super().__init__(par, v, float(atoms.get_masses().sum()), m=m)


def adaptive_isotherm(functional, pore, temperature_k, pressures_bar, solver=None, max_halvings=16, growth=1.5):
    """Adsorption branch with adaptive pressure continuation, to locate the end of a branch (turning point).

    feos' `Adsorption.adsorption_isotherm` continues from the previous profile and, after a failed point, restarts
    from the bulk density; it has no step reduction and does not report where a branch ends. Where it converges
    (e.g. the Mg-MOF-74 CO2 isotherms with AntiSymWhiteBear FMT) both give the same isotherm.

    Each target pressure is reached from the last converged profile; on failure the step in ln p is
    halved (up to `max_halvings` times). Returns dict with the converged target pressures, absolute
    loadings (mol/kg) and profiles, plus `p_end` if the branch ends before the last target (turning point).
    """
    if np.any(np.diff(pressures_bar) <= 0):
        raise ValueError("adaptive_isotherm follows the adsorption branch: pressures must be strictly increasing")
    solver = solver or feos.DFTSolver().newton(max_iter=50)
    T = temperature_k * si.KELVIN
    state = lambda p: feos.State(functional, T, pressure=p * si.BAR)  # noqa: E731
    prof = pore.initialize(state(pressures_bar[0])).solve(solver)
    p_cur, out_p, out_n, profiles, p_end = pressures_bar[0], [], [], [], None
    for p_t in pressures_bar:
        step, halvings = np.log(p_t / p_cur), 0
        while p_cur < p_t * (1 - 1e-12):
            p_new = min(p_t, p_cur * np.exp(step))
            try:
                prof = pore.initialize(state(p_new), density=prof.density).solve(solver)
                p_cur, step = p_new, step * growth
            except Exception:  # noqa: BLE001
                step, halvings = step / 2, halvings + 1
                if halvings > max_halvings:
                    p_end = p_cur
                    break
        if p_end is not None:
            break
        out_p.append(p_t); out_n.append(float(pore.per_mass(prof.total_moles))); profiles.append(prof)
    return {"pressure_bar": np.array(out_p), "mol_per_kg": np.array(out_n), "profiles": profiles, "p_end": p_end}


def canonical_isotherm(functional, pore, temperature_k, n_molecules, solver=None, density=None):
    """Isotherm traced in the number of molecules N (feos PoreSpecification.Moles): for each N the
    density profile is solved with the chemical potential as unknown; feos then updates the bulk state
    in equilibrium with the pore, which gives p(N). Unlike pressure control this passes through
    turning points of N(p), so a maximum in p(N) reveals a first-order transition of the cDFT model.

    `n_molecules`: increasing numbers of molecules per cell (single component). Returns dict with
    N per cell, pressure / bar, mol/kg and the profiles; stops at the first N that does not converge.
    """
    solver = solver or feos.DFTSolver().newton(max_iter=100)
    NAV = 6.02214076e23
    T = temperature_k * si.KELVIN
    # start from a low-pressure grand-canonical profile
    bulk = feos.State(functional, T, pressure=1e-4 * si.BAR, density_initialization="vapor")
    prof = pore.initialize(bulk, density=density).solve(solver)
    out_n, out_p, profiles, failed = [], [], [], None
    for n in n_molecules:
        spec = feos.PoreSpecification.Moles(np.array([n / NAV]) * si.MOL)
        try:
            prof = pore.initialize(prof.bulk, density=prof.density, specification=spec).solve(solver)
        except Exception as e:  # noqa: BLE001
            failed = (n, str(e)[:100])
            break
        out_n.append(float(prof.total_moles / si.MOL) * NAV)
        out_p.append(float(prof.bulk.pressure() / si.BAR))
        profiles.append(prof)
    n_arr = np.array(out_n)
    return {"molecules_per_cell": n_arr, "pressure_bar": np.array(out_p),
            "mol_per_kg": n_arr / NAV / (pore.framework_mass_amu * AMU_KG), "profiles": profiles, "failed": failed}
