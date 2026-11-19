"""Batched insertion energies with a deployed NequIP model (TorchScript, nequip 0.5.x / 0.6.x).

Same interface as feos_mlip.engine.InsertionEngine (used by compute_vext / refine_orientations):
    eng.host, eng.guest, eng.energies(guest_positions) -> U (eV, float64), eng.E_host, eng.E_guest, eng.r_max

U(config) = E(host + guest) - E(host) - E(guest, isolated), accumulated in float64 from the per-atom energies
('atomic_energy') minus the per-atom energies of the empty host and of the isolated guest.

Models used so far (2026-11-18):
  * Goeminne et al., JCTC 19, 6313 (2023), Zenodo 7904959: model_MgMOF74_{pbed3,pbets,pbembd}.pth, trained on
    intermolecular energies; adsorbate atoms have their own types (guest C -> "Co", guest O -> "Os").
  * Tayfuroglu & Keskin, JCP 163, 054704 (2025), github.com/otayfuroglu/deepMDMC: MgMOF74_CO2.pth, trained on total
    energies of cluster fragments; guest atoms use the framework types (C, O).
IMPORTANT, energy reference (checked 2026-11-19): U always subtracts E(guest, isolated). That is right for models
trained on TOTAL energies (deepMDMC). For models trained on INTERmolecular energies with a short cutoff (Goeminne,
5 A), the model energy of a CO2 with no framework atom within the cutoff (engine.E_guest, -4.3..-6.6 kJ/mol) is the
learned long-range dispersion background (their DFT data: -3.7 kJ/mol (D3), -5.1..-6.2 kJ/mol (MBD) for a single CO2
5-9 A from all atoms). For those models the correct V_ext is E(host+guest) - E(host) = U + E_guest, which is also
what Goeminne's gcmc.py uses. examples/mgmof74_co2/ext_models/ext_isotherms.py applies it as "paper+Eguest".
`guest_types` gives the model type name of every guest atom (default: its chemical symbol); host atoms always use
their chemical symbols.

NequIP's deployed models compute forces internally (autograd). The engine calls the energy network directly
(no backward pass) after checking at start-up that it reproduces the full model's per-atom energies.
"""
import math
import warnings

import numpy as np
import torch
from ase import Atoms


def _neighbors(positions, cell, r_max):
    from vesin import NeighborList
    i, j, S = NeighborList(cutoff=r_max, full_list=True).compute(points=positions, box=cell, periodic=True,
                                                                quantities="ijS")
    return np.stack([i, j]).astype(np.int64), S.astype(float)


class NequIPInsertionEngine:
    def __init__(self, model_path, host, guest, guest_types=None, device="cuda", dtype=None, guest_box=40.0,
                 n_layers=None, energy_only=True):
        self.device = torch.device(device)
        extra = {"type_names": "", "r_max": "", "num_layers": "", "nequip_version": "", "default_dtype": ""}
        self.model = torch.jit.load(model_path, map_location=self.device, _extra_files=extra).eval()
        # positions/cell in the model's default dtype (float64 for deepMDMC's MgMOF74_CO2.pth, float32 for Goeminne)
        self.dtype = dtype or (extra["default_dtype"].decode() or "float32")
        self.type_names = extra["type_names"].decode().split()
        self.r_max = float(extra["r_max"])
        self.info = {k: v.decode() for k, v in extra.items()}
        self.n_layers = n_layers or (int(extra["num_layers"]) if extra["num_layers"] else 4)

        self.host = host.copy()
        self.host.wrap()
        self.guest = guest.copy()
        self.guest_types = list(guest_types or guest.get_chemical_symbols())
        self.n_host, self.n_guest = len(self.host), len(self.guest)
        self.n_atoms = self.n_host + self.n_guest
        self.cell = np.array(self.host.cell.array, dtype=float)
        types = self.host.get_chemical_symbols() + self.guest_types
        self.types_t = torch.tensor([self.type_names.index(s) for s in types], dtype=torch.long, device=self.device)
        self._check_images()
        self.energy_only = False

        ei, S = _neighbors(self.host.positions, self.cell, self.r_max)
        self.hh = (torch.as_tensor(ei, device=self.device), torch.as_tensor(S, device=self.device, dtype=self._tdtype))
        gpos = self.guest.positions - self.guest.positions.mean(0)
        d = np.linalg.norm(gpos[:, None] - gpos[None], axis=-1)
        i, j = np.nonzero((d < self.r_max) & (d > 0))
        self.gg = torch.as_tensor(np.stack([i, j]) + self.n_host, device=self.device)

        self.guest_radius = float(np.linalg.norm(gpos, axis=1).max())
        widths = self.host.cell.volume / np.linalg.norm(np.cross(self.cell[[1, 2, 0]], self.cell[[2, 0, 1]]), axis=1)
        nr = [math.ceil((self.r_max + self.guest_radius) / w) for w in widths]
        n = np.array([(a, b, c) for a in range(-nr[0], nr[0] + 1) for b in range(-nr[1], nr[1] + 1)
                      for c in range(-nr[2], nr[2] + 1)], dtype=float)
        self.unit_t = torch.as_tensor(n, device=self.device, dtype=self._tdtype)
        self.trans = torch.as_tensor(n @ self.cell, device=self.device, dtype=self._tdtype)
        self.cell_t = torch.as_tensor(self.cell, device=self.device, dtype=self._tdtype)
        self.inv_cell_t = torch.linalg.inv(self.cell_t)
        self.host_pos_t = torch.as_tensor(self.host.positions, device=self.device, dtype=self._tdtype)

        # reference per-atom energies: empty host, isolated guest
        node_h, self.E_host = self._single(self.host, self.host.get_chemical_symbols())
        iso = Atoms(self.guest.get_chemical_symbols(), positions=gpos + guest_box / 2, cell=[guest_box] * 3, pbc=True)
        node_g, self.E_guest = self._single(iso, self.guest_types)
        self.ref_nodes = torch.cat([node_h, node_g])

        # energy-only path: validate against the full model on the empty host (per-atom energies)
        if energy_only and self._setup_energy_only():
            self.energy_only = True
            node_h2, _ = self._single(self.host, self.host.get_chemical_symbols())
            err = float((node_h2 - node_h).abs().max())
            self.energy_only = err < 1e-6 * max(1.0, float(node_h.abs().max()))
            self.info["energy_only_check_max_abs_eV"] = err
            if not self.energy_only:
                warnings.warn(f"energy-only path disagrees with the full model (max {err:.2e} eV): using the full model")

    @property
    def _tdtype(self):
        return getattr(torch, self.dtype)

    def _check_images(self):
        gpos = self.guest.positions - self.guest.positions.mean(0)
        size = 2 * np.linalg.norm(gpos, axis=1).max()
        widths = self.host.cell.volume / np.linalg.norm(np.cross(self.cell[[1, 2, 0]], self.cell[[2, 0, 1]]), axis=1)
        need = self.r_max * self.n_layers + size
        self.image_ok = widths.min() >= need
        if not self.image_ok:
            warnings.warn(f"host cell perpendicular widths {np.round(widths, 2)} A < {need:.1f} A (receptive field "
                          f"{self.n_layers} x {self.r_max} A + guest size): guest images can interact")

    def _call(self, data):
        """Per-atom energies (and total energy) of the model. Energy-only path when validated: the deployed
        wrappers (GraphModel dtype handling -> RescaleOutput -> Gradient/StressOutput -> func) are replicated without
        the autograd force/stress step; else the full model is called (with autograd)."""
        if self.energy_only:
            return self._call_energy_only(data)
        with torch.enable_grad():
            return self.model(data)

    def _setup_energy_only(self):
        m = self.model
        self._graph_model = m.original_name == "GraphModel"
        resc = m.model if self._graph_model else m
        if resc.original_name != "RescaleOutput" or not hasattr(resc.model, "func"):
            return False
        self._resc, self._func = resc, resc.model.func
        if resc.has_shift and "atomic_energy" in list(resc.shift_keys):
            return False
        self._scale = resc.scale_by if (resc.has_scale and "atomic_energy" in list(resc.scale_keys)) else None
        return True

    def _call_energy_only(self, data):
        if self._graph_model:
            mdt = self.model.model_dtype
            new = {k: (v.to(mdt) if (torch.is_floating_point(v) and k not in ("pos", "cell", "edge_cell_shift")) else v)
                   for k, v in data.items()}
            new["_model_dtype_indicator"] = self.model._model_dtype_example
        else:
            new = dict(data)
        nb = int(new["ptr"].numel() - 1) if "ptr" in new else 1
        new["cell"] = new["cell"].reshape(-1, 3, 3).expand(nb, 3, 3)
        with torch.no_grad():
            out = self._func(new)
        ae = out["atomic_energy"]
        if self._scale is not None:
            ae = ae * self._scale
        return {"atomic_energy": ae, "total_energy": ae.sum().reshape(1, 1)}

    def _single(self, atoms, type_list):
        ei, S = _neighbors(atoms.positions, atoms.cell.array, self.r_max)
        t = torch.tensor([self.type_names.index(s) for s in type_list], dtype=torch.long, device=self.device)
        data = {"pos": torch.as_tensor(atoms.positions, dtype=self._tdtype, device=self.device),
                "edge_index": torch.as_tensor(ei, device=self.device),
                "edge_cell_shift": torch.as_tensor(S, dtype=self._tdtype, device=self.device),
                "cell": torch.as_tensor(atoms.cell.array, dtype=self._tdtype, device=self.device),
                "atom_types": t}
        out = self._call(data)
        node = out["atomic_energy"].detach().double().reshape(-1)
        return node, float(out["total_energy"].detach().double().sum())

    def energies(self, guest_positions):
        """Insertion energies (eV, float64 numpy) for guest_positions (B, n_guest, 3), Cartesian A."""
        g = torch.as_tensor(np.asarray(guest_positions), device=self.device, dtype=self._tdtype)
        nb = g.shape[0]
        frac = g.mean(1, keepdim=True) @ self.inv_cell_t
        g = g - torch.floor(frac) @ self.cell_t
        n_h, n = self.n_host, self.n_atoms
        offs = torch.arange(nb, device=self.device) * n

        # guest atom i, host atom j + T: vec(i -> j) = h_j + T - g_i  => edge [i, j], shift n_T ; reverse [j, i], -n_T
        v = self.host_pos_t[None, None, :, None, :] + self.trans[None, None, None, :, :] - g[:, :, None, None, :]
        b, i, j, t = torch.nonzero((v * v).sum(-1) < self.r_max ** 2, as_tuple=True)
        gi, hj = offs[b] + n_h + i, offs[b] + j
        ei_hh = (self.hh[0][:, None, :] + offs[None, :, None]).reshape(2, -1)
        ei_gg = (self.gg[:, None, :] + offs[None, :, None]).reshape(2, -1)
        edge_index = torch.cat([ei_hh, ei_gg, torch.stack([gi, hj]), torch.stack([hj, gi])], dim=1)
        zeros_gg = torch.zeros((ei_gg.shape[1], 3), device=self.device, dtype=self._tdtype)
        shifts = torch.cat([self.hh[1].repeat(nb, 1), zeros_gg, self.unit_t[t], -self.unit_t[t]])
        pos = torch.cat([self.host_pos_t[None].expand(nb, -1, -1), g], dim=1).reshape(-1, 3)
        batch = torch.arange(nb, device=self.device).repeat_interleave(n)
        data = {"pos": pos, "edge_index": edge_index, "edge_cell_shift": shifts,
                "cell": self.cell_t[None].expand(nb, -1, -1).contiguous(),
                "atom_types": self.types_t.repeat(nb), "batch": batch,
                "ptr": torch.arange(nb + 1, device=self.device) * n}
        out = self._call(data)
        node = out["atomic_energy"].detach().double().reshape(nb, n)
        return (node - self.ref_nodes[None]).sum(1).cpu().numpy()
