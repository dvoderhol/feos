"""Diagnose FFT ringing in the FMT weighted densities of the last converged profile before each failure."""
import json
import sys
import numpy as np
import si_units as si
from ase.io import read
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import LJPore3D, MLIPPore3D, _Pore3DBase, adaptive_isotherm, pcsaft  # noqa: E402
from feos_mlip.vext import VextResult, center_on_inversion  # noqa: E402

T = 298.0
PARS = "/local/home/dvoderhol/feos/mlip-cdft/parameters/co2_pcsaft_m1.json"
par = json.load(open(PARS))[0]
co2 = pcsaft(["carbon dioxide"], PARS)
ch4 = pcsaft(["methane"], "/local/home/dvoderhol/feos/parameters/pcsaft/gross2001.json")
uc, _ = center_on_inversion(read("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif"))
mass = float(uc.get_masses().sum())
res = VextResult.load("vext_CO2_MH_production.npz")
v_ch4 = np.load("../mgmof74_ch4/v_lj_uff_0.3A.npy")
cases = [
    ("CH4 + UFF, up to 10.6 bar", ch4, lambda: _Pore3DBase(res.cellpar, np.where(v_ch4 < 50 * T, v_ch4, np.inf), mass), np.geomspace(1, 10.6, 12)),
    ("CO2 + UFF, up to 0.645 bar", co2, lambda: LJPore3D(uc, par["sigma"], par["epsilon_k"], m=1.0, spacing=0.3, cap_k=50 * T), np.geomspace(1e-3, 0.645, 15)),
    ("CO2 + MLIP, up to 0.00155 bar", co2, lambda: MLIPPore3D(res, T, spacing=0.3, framework_mass_amu=mass, m=1.0), np.geomspace(1e-5, 0.00155, 15)),
    ("CO2 + MLIP (0.2 A grid), up to 0.00155 bar", co2, lambda: MLIPPore3D(res, T, spacing=0.2, framework_mass_amu=mass, m=1.0), np.geomspace(1e-5, 0.00155, 15)),
]
for name, func, mk, ps in cases:
    pore = mk()
    iso = adaptive_isotherm(func, pore, T, ps)
    prof = iso["profiles"][-1]
    wd = prof.weighted_densities[0]          # FMT (single component, White Bear): n2, n3, n2v (3)
    n2, n3, n2v = wd[0], wd[1], wd[2:5]
    xi = np.sqrt((n2v ** 2).sum(0)) / np.where(n2 > 1e-12, n2, np.inf)
    rho = np.asarray(prof.density / (si.MOL / si.ANGSTROM ** 3))[0] * 6.02214076e23
    print(f"{name}: reached {iso['pressure_bar'][-1]:.4g} bar (end {iso['p_end']}), {iso['mol_per_kg'][-1]:.2f} mmol/g\n"
          f"   n3: min {n3.min():.2e} max {n3.max():.3f} | n2 min {n2.min():.2e} | max |n2v|/n2 {xi.max():.3f} "
          f"(points with |n2v|/n2 > 1: {(xi > 1).sum()}) | grid points with rho*sigma^3 > 5: {(rho * 3.65**3 > 5).sum()}, "
          f"max rho*sigma^3 {rho.max() * 3.65**3:.1f}", flush=True)
