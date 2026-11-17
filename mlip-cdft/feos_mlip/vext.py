"""External potential V_ext(r) on a feos periodic grid from MLIP insertion energies (energies only).

Grid: feos `Grid.periodical_3d(n, [a, b, c], [alpha, beta, gamma])` puts point (i, j, k) at the
fractional position ((i + 0.5)/n_a, (j + 0.5)/n_b, (k + 0.5)/n_c) of the unit cell. V_ext is
computed exactly at these points, for the reference atom of a rigid guest (e.g. C of CH4).

    U(r, R)    = E(host + guest(r, R)) - E(host) - E(guest)          (InsertionEngine)
    V_ext(r)   = -kT ln < exp(-U(r, R) / kT) >_R                     (Boltzmann orientation average)

The raw U(r, R) are kept, so V_ext can be re-averaged for any temperature. Points closer than
`r_ref_min` to any framework atom are not evaluated (V = inf); orientations with any guest atom
closer than `r_atom_min` to a framework atom count as overlap (exp(-U/kT) = 0). Insertion
energies below `u_min` are treated as model extrapolation and also counted as overlap.
Grid points equivalent under the space-group operations that map the grid onto itself are
computed once.
"""
import ast
import json
import time
from dataclasses import dataclass, field

import numpy as np
import spglib
from scipy.spatial.transform import Rotation

KB_EV = 8.617333262e-5


def grid_points(n):
    """Fractional coordinates of the feos periodical_3d grid, shape (n_a, n_b, n_c, 3)."""
    axes = [(np.arange(k) + 0.5) / k for k in n]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def n_points_for_spacing(atoms, spacing):
    return [int(np.ceil(length / spacing)) for length in atoms.cell.lengths()]


def irreducible_points(atoms, n, symprec=1e-3, grid_tol=0.02):
    """Map every grid point to a representative under the symmetry operations that keep the grid.

    Returns (rep_index (n_points,), irreducible indices, number of operations used).
    """
    n = np.asarray(n)
    frac = grid_points(n).reshape(-1, 3)
    rot, trans = (lambda s: (s["rotations"], s["translations"]))(spglib.get_symmetry(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=symprec))
    idx = np.arange(len(frac))
    rep = idx.copy()
    used = 0
    for w, t in zip(rot, trans):
        img = (frac @ w.T + t) % 1.0
        ijk = img * n - 0.5
        if np.abs(ijk - np.round(ijk)).max() > grid_tol:  # in grid units; structures relaxed without symmetry are symmetric only to ~1e-5
            continue  # this operation does not map the grid onto itself
        used += 1
        ijk = np.round(ijk).astype(int) % n
        j = np.ravel_multi_index(ijk.T, n)
        rep = np.minimum(rep, rep[j])
    for _ in range(10):  # propagate to the orbit minimum
        new = np.minimum(rep, rep[rep])
        if np.array_equal(new, rep):
            break
        rep = new
    return rep, np.unique(rep), used


def orientations(kind="random", n=24, seed=0):
    """Rotation matrices (n, 3, 3) applied to the guest template.

    kind:
      'random'      uniform random SO(3) rotations (fixed seed) - general molecules
      'linear'      n quasi-uniform (Fibonacci) axis directions on a hemisphere for a centrosymmetric
                    linear molecule aligned with z in its template (e.g. CO2): spin about the axis and
                    the head-tail flip are redundant
      'icosahedral' the 60 rotations of the icosahedral group (note: for CH4 only 5 of them give
                    distinct orientations, since its tetrahedral rotation group is a subgroup)
    """
    if kind == "icosahedral":
        return Rotation.create_group("I").as_matrix()
    if kind == "linear":
        k = np.arange(n) + 0.5
        cos_t = 1 - k / n                      # z in (0, 1]: upper hemisphere, equal-area spacing
        phi = np.pi * (1 + 5 ** 0.5) * k
        d = np.stack([np.sqrt(1 - cos_t ** 2) * np.cos(phi), np.sqrt(1 - cos_t ** 2) * np.sin(phi), cos_t], 1)
        rots, _ = zip(*(Rotation.align_vectors([v], [[0.0, 0.0, 1.0]]) for v in d))
        return np.array([r.as_matrix() for r in rots])
    return Rotation.random(n, random_state=seed).as_matrix()


def min_distance_to_host(points_cart, host, chunk=4096):
    """Minimum-image distance from Cartesian points to the nearest host atom."""
    cell = host.cell.array
    inv = np.linalg.inv(cell)
    out = np.empty(len(points_cart))
    for s in range(0, len(points_cart), chunk):
        d = points_cart[s:s + chunk, None, :] - host.positions[None]
        f = d @ inv
        f -= np.round(f)
        out[s:s + chunk] = np.sqrt(((f @ cell) ** 2).sum(-1)).min(1)
    return out


@dataclass
class VextResult:
    n: tuple
    cellpar: np.ndarray            # a, b, c, alpha, beta, gamma of the unit cell (feos grid)
    rep: np.ndarray                # grid point -> representative grid point
    irr: np.ndarray                # representative grid points that were evaluated
    U: np.ndarray                  # (len(irr), n_orient) insertion energies in eV, inf = overlap
    rotations: np.ndarray
    info: dict = field(default_factory=dict)

    def vext(self, temperature):
        """Orientation-averaged V_ext / K (array n_a x n_b x n_c), inf where every orientation overlaps."""
        kt = KB_EV * temperature
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            umin = np.nanmin(self.U, axis=1, keepdims=True)
            finite = np.isfinite(umin[:, 0])
            boltz = np.zeros(len(self.U))
            # nan marks orientations that are not part of a point's set (see refine_orientations)
            boltz[finite] = np.nanmean(np.exp(-(self.U[finite] - umin[finite]) / kt), axis=1)
            v_irr = np.full(len(self.U), np.inf)
            v_irr[finite] = umin[finite, 0] - kt * np.log(boltz[finite])
        full = np.full(int(np.prod(self.n)), np.inf)
        lookup = np.full(int(np.prod(self.n)), -1)
        lookup[self.irr] = np.arange(len(self.irr))
        full[:] = v_irr[lookup[self.rep]]
        return (full / KB_EV).reshape(self.n)

    def save(self, path):
        np.savez_compressed(path, n=self.n, cellpar=self.cellpar, rep=self.rep, irr=self.irr, U=self.U,
                            rotations=self.rotations, info=np.array([json.dumps(self.info, default=float)]))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        text = str(z["info"][0])
        try:
            info = json.loads(text)
        except json.JSONDecodeError:  # files written before the JSON metadata (Python dict repr)
            info = ast.literal_eval(text)
        return cls(tuple(int(k) for k in z["n"]), z["cellpar"], z["rep"], z["irr"], z["U"], z["rotations"], info)


def compute_vext(engine, unit_cell, n, rotations, ref_index=0, r_ref_min=2.0, r_atom_min=1.4,
                 u_min=-1.5, batch_size=64, use_symmetry=True, progress=True):
    """Insertion energies on the feos grid of `unit_cell` (ASE Atoms) using `engine` (its host may be a
    supercell of unit_cell with the same origin and lattice orientation)."""
    t0 = time.time()
    n = tuple(int(k) for k in n)
    if use_symmetry:
        rep, irr, n_ops = irreducible_points(unit_cell, n)
    else:
        rep = np.arange(int(np.prod(n))); irr = rep.copy(); n_ops = 1
    frac = grid_points(n).reshape(-1, 3)[irr]
    ref_cart = frac @ unit_cell.cell.array
    dist = min_distance_to_host(ref_cart, engine.host)
    todo = np.nonzero(dist >= r_ref_min)[0]

    gpos0 = engine.guest.positions - engine.guest.positions[ref_index]
    U = np.full((len(irr), len(rotations)), np.inf)
    configs = [(p, r) for p in todo for r in range(len(rotations))]
    n_extrap = 0
    for s in range(0, len(configs), batch_size):
        chunk = configs[s:s + batch_size]
        pts = np.array([c[0] for c in chunk]); rots = np.array([c[1] for c in chunk])
        g = np.einsum("bij,aj->bai", rotations[rots], gpos0) + ref_cart[pts][:, None, :]
        u = engine.energies(g)
        # guest-atom level overlap test (minimum image in the host cell)
        dmin = min_distance_to_host(g.reshape(-1, 3), engine.host).reshape(len(chunk), -1).min(1)
        bad = (dmin < r_atom_min) | (u < u_min)
        n_extrap += int(((u < u_min) & (dmin >= r_atom_min)).sum())
        u[bad] = np.inf
        U[pts, rots] = u
        if progress and (s // batch_size) % 50 == 0:
            done = s + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(configs)} insertions, {rate:.0f}/s, eta {(len(configs)-done)/rate/60:.1f} min",
                  flush=True)
    info = dict(n_grid=int(np.prod(n)), n_irreducible=len(irr), n_symmetry_ops=n_ops,
                n_evaluated_points=len(todo), n_orientations=len(rotations), n_insertions=len(configs),
                n_below_u_min=n_extrap, r_ref_min=r_ref_min, r_atom_min=r_atom_min, u_min=u_min,
                E_host=engine.E_host, E_guest=engine.E_guest, wall_s=round(time.time() - t0, 1))
    return VextResult(n, unit_cell.cell.cellpar(), rep, irr, U, rotations, info)


def center_on_inversion(atoms, symprec=1e-3):
    """Translate the structure so that an inversion centre lies at the origin (if centrosymmetric).

    The feos grid ((i + 0.5)/n) is mapped onto itself by x -> -x, so V_ext needs to be computed only
    on half of the points. Returns (shifted copy, shift in fractional coordinates or None).
    """
    sym = spglib.get_symmetry((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
    for w, t in zip(sym["rotations"], sym["translations"]):
        if np.array_equal(w, -np.eye(3, dtype=int)):
            shifted = atoms.copy()
            shifted.set_scaled_positions((atoms.get_scaled_positions() - t / 2) % 1.0)
            return shifted, -t / 2
    return atoms.copy(), None


def interpolate_periodic(beta_v, n_fine, cap=50.0):
    """Interpolate beta*V_ext from the cell-centred grid `beta_v.shape` onto the cell-centred grid `n_fine`.

    The Boltzmann factor w = exp(-beta*V) (0 where V = inf) is interpolated linearly with periodic
    wrapping and converted back, beta*V = -ln w (>= cap -> cap). Interpolating w instead of beta*V keeps
    the steep framework cores from producing spurious wells (a cubic spline of the capped beta*V rings),
    and linear interpolation keeps w positive. The Boltzmann integral (Henry coefficient) is preserved only if the
    fine grid is nested in the coarse one (n_fine = k * n_coarse): 0.4 -> 0.2 A keeps K_H to 0.03 %, while
    0.4 -> 0.3 A (non-nested) changed it by -3.5 % for CO2 in Mg-MOF-74.
    """
    from scipy.interpolate import RegularGridInterpolator

    bv = np.array(beta_v, dtype=float)
    w = np.where(np.isfinite(bv), np.exp(-np.minimum(bv, cap)), 0.0)
    w[bv >= cap] = 0.0
    n = np.array(w.shape)
    padded = np.pad(w, 1, mode="wrap")
    axes = [(np.arange(-1, k + 1) + 0.5) / k for k in n]
    interp = RegularGridInterpolator(axes, padded, method="linear")
    w_fine = interp(grid_points(n_fine).reshape(-1, 3)).reshape(tuple(n_fine))
    with np.errstate(divide="ignore"):
        return np.minimum(-np.log(w_fine), cap)


def refine_orientations(result, engine, unit_cell, rotations, u_threshold=-0.15, ref_index=0, batch_size=64,
                        r_atom_min=None, u_min=None):
    """Re-evaluate the strongly attractive grid points with a denser orientation set.

    The Boltzmann average is dominated by the deep wells, where the favourable orientations can occupy a
    narrow window that a coarse orientation set misses. Points whose best insertion energy is below
    `u_threshold` (eV) are recomputed with `rotations`; V_ext at those points then uses the dense set,
    all other points keep the original set. Returns a new VextResult (U padded with inf; per-point means
    use only the finite columns of the set that point was evaluated with)."""
    info = result.info
    r_atom_min = info.get("r_atom_min", 1.4) if r_atom_min is None else r_atom_min
    u_min = info.get("u_min", -1.5) if u_min is None else u_min
    best = np.min(result.U, axis=1)
    sel = np.nonzero(best < u_threshold)[0]
    frac = grid_points(result.n).reshape(-1, 3)[result.irr[sel]]
    ref_cart = frac @ unit_cell.cell.array
    gpos0 = engine.guest.positions - engine.guest.positions[ref_index]
    U_new = np.full((len(sel), len(rotations)), np.inf)
    configs = [(p, r) for p in range(len(sel)) for r in range(len(rotations))]
    for s in range(0, len(configs), batch_size):
        chunk = configs[s:s + batch_size]
        pts = np.array([c[0] for c in chunk]); rots = np.array([c[1] for c in chunk])
        g = np.einsum("bij,aj->bai", rotations[rots], gpos0) + ref_cart[pts][:, None, :]
        u = engine.energies(g)
        dmin = min_distance_to_host(g.reshape(-1, 3), engine.host).reshape(len(chunk), -1).min(1)
        u[(dmin < r_atom_min) | (u < u_min)] = np.inf
        U_new[pts, rots] = u
    width = max(result.U.shape[1], len(rotations))
    U = np.full((len(result.U), width), np.nan)          # nan = orientation not part of this point's set
    U[:, :result.U.shape[1]] = result.U
    U[sel] = np.nan
    U[sel, :len(rotations)] = U_new
    new = VextResult(result.n, result.cellpar, result.rep, result.irr, U, result.rotations,
                     dict(info, refined_points=int(len(sel)), refined_orientations=int(len(rotations)),
                          refine_threshold_eV=u_threshold))
    return new


def apply_overlap_guard(result, unit_cell, guest, host, r_atom_min, ref_index=0, chunk=2048):
    """Discard stored insertion energies of orientations that place any guest atom closer than `r_atom_min`
    to a framework atom (U -> inf), without new MLIP evaluations. Uses `result.rotations` (the base set;
    re-run refine_orientations with the same r_atom_min for refined points)."""
    U = np.array(result.U, dtype=float)
    gpos0 = guest.positions - guest.positions[ref_index]
    ref_cart = grid_points(result.n).reshape(-1, 3)[result.irr] @ unit_cell.cell.array
    n_rot = len(result.rotations)
    for s in range(0, len(ref_cart), chunk):
        pts = ref_cart[s:s + chunk]
        g = np.einsum("rij,aj->rai", result.rotations, gpos0)[None] + pts[:, None, None, :]   # (p, r, a, 3)
        dmin = min_distance_to_host(g.reshape(-1, 3), host).reshape(len(pts), n_rot, -1).min(2)
        block = U[s:s + chunk, :n_rot]
        block[dmin < r_atom_min] = np.inf
    info = dict(result.info, r_atom_min=r_atom_min)
    return VextResult(result.n, result.cellpar, result.rep, result.irr, U, result.rotations, info)
