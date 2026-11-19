"""Harmonic framework flexibility for MLIP cDFT.

Implements the notes "Harmonic approximation for framework flexibility in cDFT" (D. Voderholzer):

    U(u; x_1..x_N) = sum_i E0(x_i) + U_gg + 1/2 u^T H u - F^T u,      F = sum_i f(x_i)     (A1-A3)

with H the Hessian of the empty framework at its MLIP minimum R0 and f(x) = -dE(x,u)/du|_0 the force the guest
at x = (r, s) exerts on the framework atoms (feos_mlip.engine.InsertionEngine.energies_forces). Integrating out the
framework on the translation-free subspace (A5) gives

    delta(x)  = -1/2 f(x)^T H+ f(x)            one-body relaxation free energy (<= 0)
    w(x, x')  = -f(x)^T H+ f(x')               framework-mediated pair interaction

cDFT: V1 = E0 + delta enters the Boltzmann orientation average; at finite loading the basin mean field (A6, A7)

    v_w(r, s) = -f(r, s)^T ubar_k,   ubar_k = H+ (F_tot - F_k),   F_k = int_{B_k} dr ds rho(r, s) f(r, s)

is solved self-consistently around feos: feos solves the PC-SAFT density profile for a given external potential,
the outer loop updates V_eff(r) = -kT ln <exp(-beta (V1 + v_w))>_s until ubar is converged. Optionally (A8) the
first-order cumulant V_eff = V - fbar0^T ubar_k.

Conventions (same as feos_mlip.vext): coarse MLIP grid = VextResult grid (cell-centred, primitive cell), host =
supercell of that cell (H, f are for the supercell, one guest per supercell: delta contains the guest's images).
The cDFT density lives on the primitive cell and is replicated into every primitive copy of the supercell when
F_tot is formed. Symmetry-reduced points (inversion) are reconstructed with the atom permutation of the operation:
f(I x)_{perm(a)} = R f(x)_a with R = -1.
Units: eV, Angstrom; rho in molecules / A^3.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .vext import KB_EV, grid_points, orientations

NAV = 6.02214076e23


# ----------------------------------------------------------------------------------------------- framework
class HarmonicFramework:
    """Harmonic empty framework: H (3M x 3M, eV/A^2) -> pseudo-inverse H+ on the translation-free subspace.

    Checks (A5): the acoustic sum rule (H t = 0), exactly three (near-)zero eigenvalues that are translations
    and strictly positive eigenvalues on the complement. Raises if the framework is mechanically unstable."""

    def __init__(self, H, zero_tol=1e-4):
        H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
        n3 = H.shape[0]
        self.n_atoms = n3 // 3
        T = np.zeros((n3, 3))
        for a in range(3):
            T[a::3, a] = 1 / np.sqrt(self.n_atoms)
        self.T = T
        self.asr = float(np.abs(H @ T).max())                 # translation invariance of the MLIP Hessian
        ev = np.linalg.eigvalsh(H)
        P = np.eye(n3) - T @ T.T
        Hu = P @ H @ P
        evu = np.linalg.eigvalsh(Hu + T @ T.T * ev[-1])       # spectrum on U plus three copies of ev[-1]
        lam_u = np.sort(evu)[: n3 - 3]
        self.eigenvalues = ev
        self.n_zero = int((np.abs(ev) < zero_tol).sum())
        self.n_negative = int((ev < -zero_tol).sum())
        self.lambda_min = float(lam_u[0])
        if self.n_negative or self.lambda_min <= 0:
            raise ValueError(f"framework not mechanically stable on U: {self.n_negative} negative eigenvalues, "
                             f"lowest eigenvalue on U {self.lambda_min:.3e} eV/A^2")
        self.Hp = P @ np.linalg.inv(Hu + T @ T.T) @ P          # = sum_{lambda > 0} e e^T / lambda
        self.Hp = 0.5 * (self.Hp + self.Hp.T)
        self.H = H

    def summary(self):
        return dict(n_atoms=self.n_atoms, acoustic_sum_rule=self.asr, n_zero=self.n_zero, n_negative=self.n_negative,
                    lowest_eigenvalues=[float(x) for x in self.eigenvalues[:6]], lambda_min_U=self.lambda_min)

    def delta(self, f, chunk=20000):
        """delta = -1/2 f^T H+ f for f of shape (..., n_atoms, 3) (eV)."""
        f2 = np.asarray(f).reshape(-1, 3 * self.n_atoms)
        out = np.empty(len(f2))
        for s in range(0, len(f2), chunk):
            x = f2[s:s + chunk].astype(float)
            out[s:s + chunk] = -0.5 * ((x @ self.Hp) * x).sum(1)
        return out.reshape(np.shape(f)[:-2])

    def displacement(self, F):
        """Mean displacement <u> = H+ F (A) for a total force F (..., n_atoms, 3)."""
        F2 = np.asarray(F, dtype=float).reshape(-1, 3 * self.n_atoms)
        return (F2 @ self.Hp).reshape(np.shape(F))


def atom_permutation(atoms, rotation, translation, tol=1e-3):
    """perm with  rotation @ x_a + translation == x_perm[a]  (mod the cell of `atoms`), same species."""
    cell = atoms.cell.array
    inv = np.linalg.inv(cell)
    x = atoms.positions
    img = x @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
    d = (img[:, None, :] - x[None, :, :]) @ inv
    d -= np.round(d)
    dist = np.linalg.norm(d @ cell, axis=-1)
    perm = dist.argmin(1)
    err = dist[np.arange(len(x)), perm].max()
    if err > tol or len(set(perm)) != len(x) or np.any(atoms.numbers[perm] != atoms.numbers):
        raise ValueError(f"operation does not map the host onto itself (max mismatch {err:.2e} A)")
    return perm, float(err)


def permute_vectors(v, perm, sign=1.0):
    """(P v)[..., perm[a], :] = sign * v[..., a, :] for per-atom vectors v (..., n_atoms, 3)."""
    out = np.empty_like(v)
    out[..., perm, :] = sign * v
    return out


# ----------------------------------------------------------------------------------------------- coupling forces
def refined_rows(result):
    """Boolean per VextResult row: True if the row was re-evaluated with the dense set of refine_orientations."""
    n_base = len(result.rotations)
    if result.U.shape[1] <= n_base:
        return np.zeros(len(result.U), dtype=bool)
    return ~np.isnan(result.U[:, n_base:]).all(1)


def orientation_set(result, row=None, refined=None):
    """Rotation set used for VextResult row `row` (base set, or the dense set of refine_orientations)."""
    if refined is None:
        refined = bool(refined_rows(result)[row])
    if refined:
        return orientations(result.info.get("orientation_kind", "linear"), int(result.info["refined_orientations"]))
    return result.rotations


@dataclass
class FlexCoupling:
    """Coupling forces of all configurations (irreducible point row, orientation column) with E0 < e_cut."""
    row: np.ndarray            # (n_cfg,) row of VextResult.U / irr
    col: np.ndarray            # (n_cfg,) orientation column
    E0: np.ndarray             # (n_cfg,) stored insertion energy (eV)
    U_check: np.ndarray        # (n_cfg,) insertion energy recomputed with the forces (eV)
    f: np.ndarray              # (n_cfg, n_host, 3) float32 coupling forces (eV/A)
    e_cut: float
    info: dict = field(default_factory=dict)

    def save(self, path):
        import json
        np.savez(path, row=self.row, col=self.col, E0=self.E0, U_check=self.U_check, f=self.f, e_cut=self.e_cut,
                 info=np.array([json.dumps(self.info, default=float)]))

    @classmethod
    def load(cls, path):
        import json
        z = np.load(path)
        return cls(z["row"], z["col"], z["E0"], z["U_check"], z["f"], float(z["e_cut"]), json.loads(str(z["info"][0])))


def compute_coupling(result, engine, unit_cell, e_cut, ref_index=0, batch_size=32, progress=True):
    """Coupling forces f(x) for every stored configuration with E0 < e_cut (eV) of a VextResult."""
    t0 = time.time()
    U = result.U
    rows, cols = np.nonzero(np.isfinite(U) & (np.nan_to_num(U, nan=np.inf) < e_cut))
    frac = grid_points(result.n).reshape(-1, 3)[result.irr]
    ref_cart = frac @ unit_cell.cell.array
    gpos0 = engine.guest.positions - engine.guest.positions[ref_index]
    ref = refined_rows(result)[rows]
    base = orientation_set(result, refined=False)
    if np.any(cols[~ref] >= len(base)):
        raise ValueError("finite energy outside the base orientation set of a non-refined point")
    rot = base[np.minimum(cols, len(base) - 1)]
    sets = {len(base): True}
    if ref.any():
        dense = orientation_set(result, refined=True)
        rot[ref] = dense[cols[ref]]
        sets[len(dense)] = True
    f = np.empty((len(rows), engine.n_host, 3), dtype=np.float32)
    u_chk = np.empty(len(rows))
    for s in range(0, len(rows), batch_size):
        sl = slice(s, s + batch_size)
        g = np.einsum("bij,aj->bai", rot[sl], gpos0) + ref_cart[rows[sl]][:, None, :]
        u_chk[sl], f[sl] = engine.energies_forces(g)
        if progress and (s // batch_size) % 200 == 0:
            done = min(s + batch_size, len(rows)); rate = done / (time.time() - t0)
            print(f"  {done}/{len(rows)} configurations, {rate:.0f}/s, eta {(len(rows) - done) / rate / 60:.1f} min",
                  flush=True)
    info = dict(n_configurations=int(len(rows)), e_cut_eV=float(e_cut), wall_s=round(time.time() - t0, 1),
                max_abs_U_check_minus_E0=float(np.abs(u_chk - U[rows, cols]).max()),
                orientation_set_sizes=sorted(sets))
    return FlexCoupling(rows, cols, U[rows, cols].astype(float), u_chk, f, float(e_cut), info)


# ----------------------------------------------------------------------------------------------- basins
def steepest_descent_basins(v):
    """Basins of a periodic 3D grid function v (inf = inaccessible) by discrete steepest descent over the
    26 neighbours (index space). Returns (basin label per point, -1 if inaccessible; minimum index per basin)."""
    n = v.shape
    flat = v.reshape(-1)
    idx = np.arange(flat.size).reshape(n)
    best_drop = np.zeros(flat.size)
    target = np.arange(flat.size)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                nb_v = np.roll(v, (-di, -dj, -dk), axis=(0, 1, 2)).reshape(-1)
                nb_i = np.roll(idx, (-di, -dj, -dk), axis=(0, 1, 2)).reshape(-1)
                drop = (flat - nb_v) / np.sqrt(di * di + dj * dj + dk * dk)
                with np.errstate(invalid="ignore"):
                    better = np.isfinite(nb_v) & np.isfinite(flat) & (drop > best_drop)
                best_drop[better] = drop[better]
                target[better] = nb_i[better]
    for _ in range(flat.size):                      # pointer jumping to the minimum
        nxt = target[target]
        if np.array_equal(nxt, target):
            break
        target = nxt
    minima = np.unique(target[np.isfinite(flat)])
    label = np.full(flat.size, -1)
    lookup = {m: k for k, m in enumerate(minima)}
    acc = np.isfinite(flat)
    label[acc] = np.array([lookup[t] for t in target[acc]])
    return label.reshape(n), minima


# ----------------------------------------------------------------------------------------------- V_ext with delta
class FlexibleVext:
    """Orientation-resolved one-body data on the full coarse grid, with framework flexibility.

    `result` VextResult, `coupling` FlexCoupling, `framework` HarmonicFramework (of the engine host),
    `unit_cell` the (inversion-centred) primitive cell of the grid, `host` the supercell used for H and f.
    Configurations with E0 >= e_cut are inaccessible (A: the linear coupling is not valid at overlap)."""

    def __init__(self, result, coupling, framework, unit_cell, host, temperature, e_cut=None, u_cut=None):
        """u_cut (A): validity range of the harmonic/linear model. Configurations whose harmonic response
        u* = H+ f has an atomic displacement max_a |u*_a| > u_cut are treated rigidly (delta = 0 and f = 0, i.e. no
        framework coupling at all). Explicit relaxations (validate_delta_relax*.py) give relaxation/delta = 0.8-1.07
        for max|u*| < 0.2 A but only 0.3-0.6 beyond, so the harmonic delta must not be used there; treating these
        configurations rigidly underestimates their (real, smaller) relaxation. None = no cut (pure notes model)."""
        self.result, self.framework, self.T = result, framework, float(temperature)
        self.u_cut = u_cut
        self.kT = KB_EV * self.T
        self.e_cut = coupling.e_cut if e_cut is None else float(e_cut)
        if self.e_cut > coupling.e_cut + 1e-12:
            raise ValueError(f"e_cut {self.e_cut} above the cut of the computed coupling forces {coupling.e_cut}")
        keep = coupling.E0 < self.e_cut
        self.row, self.E0 = coupling.row[keep], coupling.E0[keep]
        self.f = np.ascontiguousarray(coupling.f[keep].reshape(int(keep.sum()), -1))   # (n_cfg, 3M) float32
        self.delta = framework.delta(self.f.reshape(len(self.f), -1, 3))
        self.u_star_max = np.empty(len(self.f))
        for s in range(0, len(self.f), 20000):
            u = (self.f[s:s + 20000].astype(float) @ framework.Hp).reshape(-1, framework.n_atoms, 3)
            self.u_star_max[s:s + 20000] = np.linalg.norm(u, axis=2).max(1)
        self.rigid = np.zeros(len(self.f), dtype=bool) if u_cut is None else self.u_star_max > u_cut
        self.f[self.rigid] = 0.0
        self.delta[self.rigid] = 0.0
        ref = refined_rows(result)
        self.n_set = np.full(len(result.irr), len(result.rotations))
        if ref.any():
            self.n_set[ref] = len(orientation_set(result, refined=True))
        # full grid -> (representative row, inverted?)
        n_full = int(np.prod(result.n))
        lookup = np.full(n_full, -1)
        lookup[result.irr] = np.arange(len(result.irr))
        self.grid_row = lookup[result.rep]
        self.inverted = result.rep != np.arange(n_full)
        frac = grid_points(result.n).reshape(-1, 3)
        if self.inverted.any():                         # every non-representative point is the inversion image
            d = frac[self.inverted] + frac[result.rep[self.inverted]]
            if np.abs(d - np.round(d)).max() > 1e-6:
                raise ValueError("symmetry-reduced points are not inversion images (x -> -x)")
        a, b, c = unit_cell.cell.array
        self.perm_inv, self.err_inv = atom_permutation(host, -np.eye(3), a + b + c)
        # translations of the primitive cell inside the supercell (replicated cDFT density)
        reps = np.rint(np.linalg.solve(unit_cell.cell.array.T, host.cell.array.T).diagonal()).astype(int)
        self.translations = []
        for i in range(reps[0]):
            for j in range(reps[1]):
                for k in range(reps[2]):
                    if i or j or k:
                        self.translations.append(atom_permutation(host, np.eye(3), i * a + j * b + k * c)[0])
        # configurations of each full grid point: cfg ids grouped by row
        order = np.argsort(self.row, kind="stable")
        starts = np.searchsorted(self.row[order], np.arange(len(result.irr) + 1))
        self.cfg_of_row = [order[starts[r]:starts[r + 1]] for r in range(len(result.irr))]
        # expanded (grid point, cfg) pairs for vectorised sums
        pts, cfg = [], []
        for j in range(n_full):
            r = self.grid_row[j]
            if r >= 0 and len(self.cfg_of_row[r]):
                c = self.cfg_of_row[r]
                pts.append(np.full(len(c), j)); cfg.append(c)
        self.pair_point = np.concatenate(pts)
        self.pair_cfg = np.concatenate(cfg)
        self.pair_inv = self.inverted[self.pair_point]
        self.pair_nset = self.n_set[self.grid_row[self.pair_point]]
        self.n_full = n_full
        self.cell_volume = abs(np.linalg.det(unit_cell.cell.array))
        self.dV = self.cell_volume / n_full

    # -- one-body potentials ------------------------------------------------------------------------
    def _average(self, energy_pair):
        """-kT ln[(1/n_set) sum_accessible exp(-beta e)] per grid point (eV, inf if nothing accessible),
        and the normalised orientation weights pi per pair."""
        e = energy_pair
        emin = np.full(self.n_full, np.inf)
        np.minimum.at(emin, self.pair_point, e)
        x = np.exp(-(e - emin[self.pair_point]) / self.kT)
        s = np.zeros(self.n_full)
        np.add.at(s, self.pair_point, x)
        v = np.full(self.n_full, np.inf)
        ok = s > 0
        nset = np.zeros(self.n_full)
        nset[self.pair_point] = self.pair_nset
        v[ok] = emin[ok] - self.kT * np.log(s[ok] / nset[ok])
        pi = x / s[self.pair_point]
        return v, pi

    def v_rigid(self):
        """V(r) in K without flexibility (E0 only, configurations above e_cut removed)."""
        return self._average(self.E0[self.pair_cfg])[0].reshape(self.result.n) / KB_EV

    def v_flex(self):
        """V(r) in K with the one-body relaxation free energy delta (infinite dilution, Eq. Vr)."""
        return self._average((self.E0 + self.delta)[self.pair_cfg])[0].reshape(self.result.n) / KB_EV

    # -- mean field ---------------------------------------------------------------------------------
    def basins(self):
        return steepest_descent_basins(self.v_flex())

    def v_w(self, ubar, label):
        """v_w per (grid point, configuration) pair: -g . ubar_k with g = f (representative point) or the
        inversion image g = -P f, for which -g . u = f . u[perm] (P: (P v)[perm[a]] = v[a])."""
        K = len(ubar)
        U = ubar.reshape(K, -1)
        U_inv = -ubar[:, self.perm_inv, :].reshape(K, -1)          # -u[perm]:  -g.u = -f.(-u[perm])
        G = self.f @ np.concatenate([U, U_inv]).T.astype(np.float32)   # (n_cfg, 2K)
        k = label.reshape(-1)[self.pair_point]
        return -G[self.pair_cfg, k + K * self.pair_inv].astype(float)

    def basin_forces(self, rho_coarse, pi, label):
        """F_k (K, n_host, 3) of the guests in basin k of the primitive cell (copy 0 of the supercell)."""
        from scipy.sparse import csr_matrix
        k = label.reshape(-1)[self.pair_point]
        K = int(label.max()) + 1
        wgt = rho_coarse.reshape(-1)[self.pair_point] * pi * self.dV
        W = csr_matrix((wgt, (k + K * self.pair_inv, self.pair_cfg)), shape=(2 * K, len(self.f)))
        FF = np.asarray(W @ self.f.astype(float))                 # (2K, 3M): direct | inverted (raw f sums)
        F_dir = FF[:K].reshape(K, -1, 3)
        F_inv = permute_vectors(FF[K:].reshape(K, -1, 3), self.perm_inv, -1.0)
        return F_dir + F_inv

    def total_force(self, Fk):
        """F_tot over the supercell: every primitive copy carries the same (replicated) density."""
        F = Fk.sum(0)
        return F + sum(permute_vectors(F, p) for p in self.translations)

    def mean_field(self, rho_coarse, ubar, label, cumulant=False):
        """One evaluation of the mean field: returns (V_eff per full grid point in eV, new ubar, F_k, pi)."""
        e1 = (self.E0 + self.delta)[self.pair_cfg]
        if cumulant:                                     # (A8): orientation weights from V1 only
            v1, pi0 = self._average(e1)
            fbar_vw = np.zeros(self.n_full)
            np.add.at(fbar_vw, self.pair_point, pi0 * self.v_w(ubar, label))
            v_eff, pi = v1 + fbar_vw, pi0
        else:
            v_eff, pi = self._average(e1 + self.v_w(ubar, label))
        Fk = self.basin_forces(rho_coarse, pi, label)
        Ft = self.total_force(Fk)
        u_new = self.framework.displacement(Ft[None] - Fk)
        return v_eff, u_new, Fk, pi

    def free_energy_w(self, rho_coarse, label, pi):
        """F_w = -1/2 sum_k sum_{l != k} F_k^T H+ F_l over the supercell (eV), for derivative checks."""
        Fk = self.basin_forces(rho_coarse, pi, label)
        allF = [Fk] + [permute_vectors(Fk, p) for p in self.translations]
        allF = np.concatenate(allF).reshape(-1, self.f.shape[1])
        Ft = allF.sum(0)
        q = allF @ self.framework.Hp
        return -0.5 * (Ft @ self.framework.Hp @ Ft - np.einsum("ki,ki->", q, allF))


# ----------------------------------------------------------------------------------------------- feos coupling
class FlexibleMLIPPore3D:
    """feos pore (periodic 3D, PC-SAFT) with V_ext from MLIP insertion energies + harmonic framework flexibility.

    `.grid`, `.external_potential` (infinite dilution: V with delta) mirror feos.Pore1D / MLIPPore3D.
    `solve(bulk)` runs the self-consistent basin mean field (finite loading) and returns the converged
    feos PoreProfile together with the mean framework displacement."""

    def __init__(self, flexvext, n_grid=None, cap=50.0, framework_mass_amu=None, m=1.0, cumulant=False):
        from .cdft import periodic_grid, external_potential  # noqa: F401
        self.fv = flexvext
        self.T = flexvext.T
        self.n_coarse = np.array(flexvext.result.n)
        self.n_fine = np.array(n_grid) if n_grid is not None else 2 * self.n_coarse
        if np.any(self.n_fine % self.n_coarse):
            raise ValueError("the feos grid must be nested in the MLIP grid (n_feos = k * n_MLIP)")
        self.k = self.n_fine // self.n_coarse
        self.cap = cap
        self.m = m
        self.cumulant = cumulant
        self.cellpar = np.asarray(flexvext.result.cellpar, dtype=float)
        self.framework_mass_amu = framework_mass_amu or flexvext.result.info.get("framework_mass_amu")
        self.grid = periodic_grid(self.cellpar, self.n_fine)
        self.label, self.minima = flexvext.basins()
        self.n_basins = len(self.minima)
        self.external_potential = self._potential(flexvext.v_flex())

    def _potential(self, v_coarse_k):
        from .cdft import external_potential
        from .vext import interpolate_periodic
        bv = interpolate_periodic(np.asarray(v_coarse_k).reshape(tuple(self.n_coarse)) / self.T, tuple(self.n_fine),
                                  cap=self.cap)
        return external_potential(np.where(bv >= self.cap, np.inf, bv * self.T))

    def coarse_density(self, profile):
        """feos density (mol/m^3 on the fine grid) -> molecules/A^3 averaged over the fine cells of each coarse cell."""
        import si_units as si
        rho = np.asarray(profile.density / (si.MOL / si.METER ** 3))[0] * NAV * 1e-30
        ka, kb, kc = self.k
        na, nb, nc = self.n_coarse
        return rho.reshape(na, ka, nb, kb, nc, kc).mean(axis=(1, 3, 5))

    def henry_coefficient(self, temperature_k=None):
        from .cdft import henry_from_vext
        if self.m != 1.0:
            raise ValueError("Henry coefficient from V_ext is only defined for m = 1")
        return henry_from_vext(self.fv.v_flex(), self.cellpar, temperature_k or self.T, self.framework_mass_amu)

    def per_mass(self, moles):
        import si_units as si
        from .cdft import AMU_KG
        return np.asarray(moles / si.MOL) / (self.framework_mass_amu * 1.0 * AMU_KG)

    def solve(self, bulk, density=None, ubar=None, solver=None, mixing=0.5, tol_u=1e-5, max_iter=200,
              verbose=False):
        """Self-consistent finite-loading solution at the bulk state `bulk` (feos State).

        Outer loop: V_eff(ubar) -> feos profile -> rho -> F_k -> ubar_new = H+ (F_tot - F_k); damped Picard
        on ubar (`mixing`), until max|ubar_new - ubar| < tol_u (A). Returns dict(profile, ubar, history,
        u_mean (mean framework displacement, A), F_k)."""
        import feos
        solver = solver or feos.DFTSolver().newton(max_iter=100)
        fv = self.fv
        ubar = np.zeros((self.n_basins, fv.framework.n_atoms, 3)) if ubar is None else np.array(ubar)
        hist = []
        prof = None
        for it in range(max_iter):
            e1 = fv.mean_field(np.zeros(self.n_coarse), ubar, self.label, self.cumulant)[0] if it == 0 else v_eff
            pot = self._potential(e1 / KB_EV)
            prof = feos.PoreProfile(self.grid, bulk, pot, density=density).solve(solver)
            density = prof.density
            rho = self.coarse_density(prof)
            v_eff, u_new, Fk, pi = fv.mean_field(rho, ubar, self.label, self.cumulant)
            du = float(np.abs(u_new - ubar).max())
            n_tot = float(rho.sum() * fv.dV)
            hist.append((du, n_tot))
            if verbose:
                print(f"    it {it:3d}  max|du| {du:.2e} A  N/cell {n_tot:.5f}", flush=True)
            if du < tol_u:
                ubar = u_new
                break
            ubar = (1 - mixing) * ubar + mixing * u_new
            v_eff = fv.mean_field(rho, ubar, self.label, self.cumulant)[0]
        else:
            raise RuntimeError(f"mean field not converged after {max_iter} iterations (max|du| {du:.2e} A)")
        # final profile with the converged ubar
        v_eff, _, Fk, pi = fv.mean_field(rho, ubar, self.label, self.cumulant)
        prof = feos.PoreProfile(self.grid, bulk, self._potential(v_eff / KB_EV), density=density).solve(solver)
        rho = self.coarse_density(prof)
        Fk = fv.basin_forces(rho, fv.mean_field(rho, ubar, self.label, self.cumulant)[3], self.label)
        u_mean = fv.framework.displacement(fv.total_force(Fk))
        return dict(profile=prof, ubar=ubar, history=hist, u_mean=u_mean, F_k=Fk, iterations=len(hist))
