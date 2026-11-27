"""Single-guest configurations per adsorption basin, from a stored V_ext (VextResult), mapped to the full unit cell.

Basins are those of the notes (Sec. 'Finite loading'): steepest descent on the orientation-averaged V (flex.
steepest_descent_basins). A configuration is (position of the reference atom, axis of a linear guest, rigid U in eV).
The stored insertion energies are on irreducible grid points; a configuration on an equivalent point is obtained by
the space-group operation that maps the representative onto it (same grid tolerance as vext.irreducible_points).
"""
import numpy as np
import spglib

from .vext import grid_points, orientations

KB_EV = 8.617333262e-5
EV_J = 1.602176634e-19


class BasinConfigs:
    def __init__(self, result, unit_cell, temperature):
        from .flex import steepest_descent_basins
        self.res = result
        self.uc = unit_cell
        self.T = temperature
        self.label, self.minima = steepest_descent_basins(result.vext(temperature))
        self.label = self.label.reshape(-1)
        self.n_basins = len(self.minima)
        n_base = len(result.rotations)
        U = result.U
        self.refined = (~np.all(np.isnan(U[:, n_base:]), axis=1)) if U.shape[1] > n_base else np.zeros(len(U), bool)
        self.rot192 = orientations("linear", 192)
        self.lookup = np.full(int(np.prod(result.n)), -1)
        self.lookup[result.irr] = np.arange(len(result.irr))
        self.frac = grid_points(result.n).reshape(-1, 3)
        sym = spglib.get_symmetry((unit_cell.cell.array, unit_cell.get_scaled_positions(), unit_cell.numbers),
                                  symprec=1e-3)
        self.ops = list(zip(sym["rotations"], sym["translations"]))
        self.A = unit_cell.cell.array.T

    def _rows(self, k):
        """(full-grid points of basin k, their irreducible rows, number of orientations per row)."""
        pts = np.nonzero(self.label == k)[0]
        rows = self.lookup[self.res.rep[pts]]
        nrot = np.where(self.refined[rows], 192, len(self.res.rotations))
        return pts, rows, nrot

    def config(self, point, col):
        """(reference-atom position (A), unit axis, U (eV)) of orientation `col` at full-grid point `point`."""
        row = self.lookup[self.res.rep[point]]
        rotset = self.rot192 if self.refined[row] else self.res.rotations
        axis = rotset[col] @ np.array([0.0, 0.0, 1.0])
        f0, f1 = self.frac[self.res.irr[row]], self.frac[point]
        for W, t in self.ops:
            d = (W @ f0 + t) - f1
            if np.all(np.abs((d - np.round(d)) * np.asarray(self.res.n)) < 0.02):
                axis = self.A @ W @ np.linalg.inv(self.A) @ axis
                return self.A @ f1, axis / np.linalg.norm(axis), float(self.res.U[row, col])
        raise RuntimeError("no symmetry operation maps the representative onto the grid point")

    def minimum(self, k):
        """Lowest-U configuration (over points and orientations) of basin k."""
        pts, rows, nrot = self._rows(k)
        best = (np.inf, None, None)
        for p, r, n in zip(pts, rows, nrot):
            u = self.res.U[r, :n]
            if np.isfinite(u).any():
                c = int(np.nanargmin(np.where(np.isfinite(u), u, np.nan)))
                if u[c] < best[0]:
                    best = (u[c], p, c)
        return self.config(best[1], best[2])

    def sample(self, k, n, rng):
        """n Boltzmann samples (weight exp(-U/kT) per point and orientation, uniform orientation measure)."""
        pts, rows, nrot = self._rows(k)
        P, C, w = [], [], []
        for p, r, m in zip(pts, rows, nrot):
            u = self.res.U[r, :m]
            ok = np.isfinite(u)
            P += [p] * int(ok.sum()); C += list(np.nonzero(ok)[0]); w += list(u[ok])
        w = np.array(w)
        w = np.exp(-(w - w.min()) / (KB_EV * self.T)) / nrot[np.searchsorted(pts, P)]   # pts is sorted
        w /= w.sum()
        pick = rng.choice(len(w), size=n, p=w)
        return [self.config(P[j], C[j]) for j in pick]


# ------------------------------------------------------------------ framework-mediated guest-guest interaction (W)
def basin_pairs(bc, r_cut):
    """Basin pairs (k, l, T) whose minima are closer than r_cut (A); T = integer lattice translation of basin l.
    Each unordered pair once (W_kl(T) = W_lk(-T)); the pair (k, k, 0) is excluded (at most one guest per basin)."""
    P = bc.A @ bc.frac[bc.minima].T
    P = P.T
    shifts = np.array([(i, j, k) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) for k in (-2, -1, 0, 1, 2)])
    out = []
    for k in range(bc.n_basins):
        for l in range(k, bc.n_basins):
            for T in shifts:
                if l == k and (tuple(T) <= (0, 0, 0)):      # (k,k,0) excluded; (k,k,T) and (k,k,-T) counted once
                    continue
                d = np.linalg.norm(P[l] + bc.A @ T - P[k])
                if d < r_cut:
                    out.append((k, l, tuple(int(x) for x in T), float(d)))
    return out


def basin_pair_interaction(bc, pairs, energy, host, c_o, n_samples, rng, min_atom_dist=2.6, box=40.0, kT=None):
    """Static framework-mediated pair interaction per basin pair (eV), from
        u_pore = E(fw+A+B) - E(fw+A) - E(fw+B) + E(fw),   u_vac = E(A+B) - E(A) - E(B)   (no framework)
    for A, B independent Boltzmann samples of basins k and l (+T):
        W      = -kT ln( <exp(-u_pore/kT)> / <exp(-u_vac/kT)> )   in-pore minus vacuum pair free energy (used)
        W_mean = <u_pore - u_vac>                                    first-order mean (dominated by rare
                                                                     extreme samples; stored for comparison)
    Samples with an intermolecular atom distance < min_atom_dist are rejected; if fewer than half of the attempted
    samples are accepted the basins are mutually exclusive ('exclusive': True, W = 0: their exclusion is the hard-core
    part of the fluid functional). `energy(atoms)` returns the potential energy (eV) of an ASE Atoms object (guest
    atoms carry tag 1). Per-sample u_pore, u_vac are stored."""
    from ase import Atoms

    def mol(c, ax):
        return np.array([c, c + c_o * ax, c - c_o * ax])

    def with_guests(base, ms, cell, pbc=True):
        g = Atoms("CO2" * len(ms), positions=np.concatenate(ms), cell=cell, pbc=pbc)
        g.set_tags([1] * len(g))
        if base is None:
            return g
        a = base.copy()
        a.set_tags([0] * len(a))
        return a + g

    e_fw = energy(host)
    centre = np.full(3, box / 2)
    e_mol = energy(with_guests(None, [mol(centre, np.array([0.0, 0.0, 1.0]))], np.eye(3) * box))
    out = []
    kT = kT if kT is not None else KB_EV * bc.T
    for k, l, T, d in pairs:
        sa, sb = bc.sample(k, 3 * n_samples, rng), bc.sample(l, 3 * n_samples, rng)
        shift = bc.A @ np.array(T, dtype=float)
        u_p, u_v, tried = [], [], 0
        for (ca, xa, _), (cb, xb, _) in zip(sa, sb):
            if len(u_p) == n_samples:
                break
            tried += 1
            A, B = mol(ca, xa), mol(cb + shift, xb)
            if np.min(np.linalg.norm(A[:, None] - B[None], axis=-1)) < min_atom_dist:
                continue
            e_ab = energy(with_guests(host, [A, B], host.cell))
            e_a = energy(with_guests(host, [A], host.cell))
            e_b = energy(with_guests(host, [B], host.cell))
            cv = centre - ca
            u_v.append(energy(with_guests(None, [A + cv, B + cv], np.eye(3) * box)) - 2 * e_mol)
            u_p.append(e_ab - e_a - e_b + e_fw)
        u_p, u_v = np.array(u_p), np.array(u_v)
        exclusive = len(u_p) < 0.5 * tried
        if len(u_p) and not exclusive:
            lse = lambda x: -kT * (np.log(np.mean(np.exp(-(x - x.min()) / kT))) - x.min() / kT)  # noqa: E731
            w_free = float(lse(u_p) - lse(u_v))
        else:
            w_free = 0.0
        out.append(dict(k=k, l=l, T=list(T), d_min=d, n=len(u_p), tried=tried, exclusive=bool(exclusive), W=w_free,
                        W_mean=float((u_p - u_v).mean()) if len(u_p) else 0.0,
                        u_pore_samples=u_p.tolist(), u_vac_samples=u_v.tolist()))
    return out


class BasinPairField:
    """Basin pair interaction (notes, Sec. 'Finite loading', with W in place of / in addition to the elastic term):
    a guest in basin k feels v_k = sum_(l,T) W_kl(T) theta_l, theta_l = mean number of guests in basin l per unit cell
    (uncorrelated basin occupancies, at most one guest per basin, (A7)). In cDFT this is the feos RegionMeanField
    F_mf = 1/2 sum_kl W_kl theta_k theta_l with the basins as regions (`mean_field`)."""

    def __init__(self, entries, n_basins):
        self.W = np.zeros((n_basins, n_basins))
        for e in entries:                       # sum over translations; W_kl(T) and W_lk(-T) both act
            if e.get("exclusive"):
                continue
            k, l, w = e["k"], e["l"], e["W"]
            if k == l:
                self.W[k, k] += 2 * w           # (k,k,T) and (k,k,-T)
            else:
                self.W[k, l] += w
                self.W[l, k] += w

    def mean_field(self, label_coarse, n_grid):
        """feos.RegionMeanField on a feos grid with `n_grid` points nested in the grid of `label_coarse` (basin label
        on the MLIP grid, -1 = inaccessible): fine points k*c .. k*c+k-1 lie in coarse cell c."""
        import feos
        import si_units as si
        label = label_coarse
        for ax, n in enumerate(n_grid):
            if n % label.shape[ax]:
                raise ValueError(f"feos grid {tuple(n_grid)} is not nested in the MLIP grid {label_coarse.shape}")
            label = np.repeat(label, n // label.shape[ax], axis=ax)
        return feos.RegionMeanField(label.astype(np.int64), self.W * EV_J * si.JOULE)
