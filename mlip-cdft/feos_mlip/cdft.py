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


def pcsaft(components, parameter_file):
    return feos.HelmholtzEnergyFunctional.pcsaft(feos.Parameters.from_json(components, parameter_file))


def henry_from_vext(vext_k, cellpar, temperature, framework_mass_amu):
    """K_H in mol/(kg Pa) directly from V_ext: K_H = <exp(-V/kT)>_cell * V_cell / (m_framework R T)."""
    from ase.cell import Cell
    vol_m3 = Cell.fromcellpar(cellpar).volume * 1e-30
    with np.errstate(over="ignore"):
        boltz = np.mean(np.exp(-np.where(np.isfinite(vext_k), vext_k, np.inf) / temperature))
    return boltz * vol_m3 / (framework_mass_amu * AMU_KG * 8.31446262 * temperature)


def isotherm(functional, vext_k, cellpar, temperature_k, pressures_bar, framework_mass_amu,
             solver=None, branch="adsorption"):
    """Pure-component isotherm. Returns dict with pressure/bar, absolute loading in molecules/cell and mol/kg."""
    grid = periodic_grid(cellpar, vext_k.shape)
    pot = external_potential(vext_k)
    p = np.asarray(pressures_bar, dtype=float) * si.BAR
    run = {"adsorption": feos.Adsorption.adsorption_isotherm,
           "desorption": feos.Adsorption.desorption_isotherm,
           "equilibrium": feos.Adsorption.equilibrium_isotherm}[branch]
    iso = run(functional, temperature_k * si.KELVIN, p, grid, pot, solver=solver or feos.DFTSolver().newton())
    n_mol = np.asarray(iso.total_adsorption / si.MOL)
    mass_kg = framework_mass_amu * AMU_KG
    return {"pressure_bar": np.asarray(iso.pressure / si.BAR), "molecules_per_cell": n_mol * 6.02214076e23,
            "mol_per_kg": n_mol / mass_kg, "raw": iso}


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
    """Adsorption branch with adaptive pressure continuation (stand-in for a fix in feos' Rust
    `Adsorption::isotherm`, which restarts from the bulk density after a failed point).

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
