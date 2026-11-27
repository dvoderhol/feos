"""Standard solid-fluid potential (baseline): Lennard-Jones 12-6 sum over framework atoms.

Same model as feos' former `external_potential_3d` (crates/feos-dft/src/adsorption/pore3d.rs before #376):
    V_i(r) = m_i * sum_alpha 4 eps_ia [(sigma_ia/r)^12 - (sigma_ia/r)^6],   r < r_c (no tail correction)
with Lorentz-Berthelot combining rules, but for any (oblique) periodic cell: all atom images within
r_c of the grid points are found with a KD-tree instead of the minimum-image convention.
"""
import numpy as np
from scipy.spatial import cKDTree

# UFF (Rappe et al. 1992): x_i [A], D_i [kcal/mol]; sigma = x 2^(-1/6), eps/k = D * 503.22 K.
# Typed from memory for the elements needed so far - verify before production use.
UFF = {"H": (2.886, 0.044), "C": (3.851, 0.105), "N": (3.660, 0.069), "O": (3.500, 0.060),
       "Mg": (3.021, 0.111), "Zn": (2.763, 0.124), "Cu": (3.495, 0.005)}


def lj_external_potential(points_cart, atoms, sigma_f, epsilon_k_f, m=1.0, cutoff=14.0,
                          forcefield=UFF, chunk=20000):
    """V_ext / K at Cartesian points (Angstrom) for a fluid segment (sigma_f [A], epsilon_k_f [K]).

    `atoms` is the periodic framework (ASE Atoms); the points must be expressed in the same frame.
    """
    sym = atoms.get_chemical_symbols()
    sig_s = np.array([forcefield[s][0] * 2 ** (-1 / 6) for s in sym])
    eps_s = np.array([forcefield[s][1] * 503.22 for s in sym])
    cell = atoms.cell.array
    widths = atoms.cell.volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    nr = np.ceil(cutoff / widths).astype(int) + 1
    shifts = np.array([(i, j, k) for i in range(-nr[0], nr[0] + 1) for j in range(-nr[1], nr[1] + 1)
                       for k in range(-nr[2], nr[2] + 1)]) @ cell
    cloud = (atoms.positions[None] + shifts[:, None]).reshape(-1, 3)
    kind = np.tile(np.arange(len(atoms)), len(shifts))
    sig2 = (0.5 * (sig_s + sigma_f)) ** 2
    eps4 = 4 * np.sqrt(eps_s * epsilon_k_f)
    tree = cKDTree(cloud)
    v = np.zeros(len(points_cart))
    for s in range(0, len(points_cart), chunk):
        pts = points_cart[s:s + chunk]
        sdm = cKDTree(pts).sparse_distance_matrix(tree, cutoff, output_type="coo_matrix")
        a = kind[sdm.col]
        r2 = np.maximum(sdm.data, 1e-6) ** 2
        sr6 = (sig2[a] / r2) ** 3
        np.add.at(v, s + sdm.row, eps4[a] * (sr6 * sr6 - sr6))
    return m * v
