"""Batched, energy-only insertion energies with a MACE model (uMOF-MH, uMOF-POLAR).

U(config) = E(host + guest) - E(host) - E(guest, isolated)

The host (a rigid framework, usually a supercell so that periodic images of the guest are
further apart than the model's receptive field) is fixed. Host-host and intramolecular guest
edges are computed once; for every batch only the guest-host edges are built, on the GPU.
The model is called with compute_force=False under no_grad, so no backward pass is done.

U is accumulated from per-atom energy differences (node_energy - reference node_energy) in
float64, which keeps float32 models accurate (total energies of ~1e3 eV would otherwise lose
~1e-4 eV in the subtraction). Models whose node energies do not add up to the total energy
(checked at start-up) fall back to total-energy differences.
"""
import copy
import math
import sys
import warnings

import numpy as np
import torch
from ase import Atoms
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools

sys.path.insert(0, "/local/home/dvoderhol/uMOF/umof_benchmark/scripts")
from mace_calc_helper import make_calc, prep_atoms_for_polar  # noqa: E402


class InsertionEngine:
    def __init__(self, model_path, host, guest, head="r2scan+d4", dtype="float32", device="cuda",
                 cueq=True, guest_box=30.0, charge=0, spin=1):
        self.device = torch.device(device)
        self.dtype = dtype
        calc, self.is_polar = make_calc(model_path, head, dtype, device)
        self.calc = calc
        self.model = calc.models[0].eval()
        if cueq:
            try:
                from mace.cli.convert_e3nn_cueq import run as e3nn_to_cueq
                self.model = e3nn_to_cueq(copy.deepcopy(self.model), device=device).to(
                    device=self.device, dtype=getattr(torch, dtype)).eval()
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"cuEquivariance conversion failed ({type(e).__name__}: {e}); using e3nn")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.r_max = float(calc.r_max)
        self.charge, self.spin = charge, spin

        self.host = host.copy()
        self.host.wrap()
        self.guest = guest.copy()
        self.n_host, self.n_guest = len(self.host), len(self.guest)
        self.n_atoms = self.n_host + self.n_guest
        self.cell = np.array(self.host.cell.array, dtype=float)
        self._check_images(guest)

        # host-host and intramolecular guest edges (fixed)
        ei, sh, us, _ = mace_data.neighborhood.get_neighborhood(
            self.host.positions, cutoff=self.r_max, pbc=(True, True, True), cell=self.cell)
        self.hh = (torch.as_tensor(ei, device=self.device),
                   torch.as_tensor(sh, device=self.device, dtype=self._tdtype),
                   torch.as_tensor(us, device=self.device, dtype=self._tdtype))
        gpos = self.guest.positions - self.guest.positions.mean(0)
        d = np.linalg.norm(gpos[:, None] - gpos[None], axis=-1)
        i, j = np.nonzero((d < self.r_max) & (d > 0))
        self.gg = torch.as_tensor(np.stack([i, j]) + self.n_host, device=self.device)

        # lattice translations needed for guest-host edges: host atoms are wrapped into the cell,
        # the guest is wrapped as a whole (by its centroid), so its atoms stick out by <= its radius
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

        self._templates = {}
        node_h, self.E_host, ok_h = self._reference_node_energies(self.host)
        iso = Atoms(self.guest.get_chemical_symbols(), positions=gpos, cell=[guest_box] * 3, pbc=True)
        iso.positions += guest_box / 2
        node_g, self.E_guest, ok_g = self._reference_node_energies(iso)
        self.ref_nodes = torch.cat([node_h, node_g])
        self.use_node_energies = ok_h and ok_g
        if not self.use_node_energies:
            warnings.warn("node energies do not add up to the total energy (e.g. long-range electrostatics): "
                          "using total-energy differences, which lose precision in float32")

    @property
    def _tdtype(self):
        return getattr(torch, self.dtype)

    def _check_images(self, guest):
        """Periodic images of the guest must not see each other through the model."""
        gpos = guest.positions - guest.positions.mean(0)
        size = 2 * np.linalg.norm(gpos, axis=1).max()
        cell = np.array(self.host.cell.array)
        widths = self.host.cell.volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
        need = 2 * self.r_max * int(self.model.num_interactions) / 2 + size  # receptive field + guest size
        if widths.min() < need:
            warnings.warn(f"host cell perpendicular widths {np.round(widths, 2)} A < {need:.1f} A "
                          f"(receptive field + guest size): guest images interact; use a larger supercell")

    def _atomic_data(self, atoms):
        prep_atoms_for_polar(atoms, self.is_polar, charge=self.charge, spin=self.spin)
        keyspec = mace_data.KeySpecification(info_keys=self.calc.info_keys, arrays_keys=self.calc.arrays_keys)
        with torch_tools.default_dtype(self.dtype):
            cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=self.calc.head)
            return mace_data.AtomicData.from_config(cfg, z_table=self.calc.z_table, cutoff=self.r_max,
                                                    heads=self.calc.available_heads)

    def _forward(self, batch_dict):
        with torch_tools.default_dtype(self.dtype), torch.no_grad():
            return self.model(batch_dict, compute_force=False, compute_stress=False, training=False)

    def _reference_node_energies(self, atoms):
        data = self._atomic_data(atoms.copy())
        batch = next(iter(torch_geometric.dataloader.DataLoader([data], batch_size=1))).to(self.device)
        out = self._forward(batch.to_dict())
        node = out["node_energy"].double()
        total = float(out["energy"].double()[0])
        return node, total, abs(float(node.sum()) - total) < 1e-3

    def _template(self, nb):
        if nb not in self._templates:
            combo = self.host + self.guest
            combo.set_cell(self.cell)
            combo.pbc = True
            data = self._atomic_data(combo)
            loader = torch_geometric.dataloader.DataLoader([data] * nb, batch_size=nb)
            self._templates[nb] = next(iter(loader)).to(self.device).to_dict()
        return self._templates[nb]

    def energies(self, guest_positions):
        """Insertion energies (eV, float64 numpy) for guest_positions of shape (B, n_guest, 3), Cartesian A."""
        g = torch.as_tensor(np.asarray(guest_positions), device=self.device, dtype=self._tdtype)
        nb = g.shape[0]
        centroid = g.mean(1, keepdim=True)
        frac = centroid @ self.inv_cell_t
        g = g - torch.floor(frac) @ self.cell_t  # wrap the molecule as a whole (keeps intramolecular vectors)
        n_h, n_g, n = self.n_host, self.n_guest, self.n_atoms
        offs = torch.arange(nb, device=self.device) * n

        # guest-host pairs within r_max: v = g_i - h_j - T  (sender host j, receiver guest i, shift -T)
        v = g[:, :, None, None, :] - self.host_pos_t[None, None, :, None, :] - self.trans[None, None, None, :, :]
        b, i, j, t = torch.nonzero((v * v).sum(-1) < self.r_max ** 2, as_tuple=True)
        recv_g, send_h = offs[b] + n_h + i, offs[b] + j
        shift_gh, ushift_gh = -self.trans[t], -self.unit_t[t]

        ei_hh, sh_hh, us_hh = self.hh
        ei_hh = (ei_hh[:, None, :] + offs[None, :, None]).reshape(2, -1)
        ei_gg = (self.gg[:, None, :] + offs[None, :, None]).reshape(2, -1)
        n_hh, n_gg = self.hh[0].shape[1], self.gg.shape[1]
        zeros_gg = torch.zeros((n_gg * nb, 3), device=self.device, dtype=self._tdtype)
        edge_index = torch.cat([ei_hh, ei_gg, torch.stack([send_h, recv_g]), torch.stack([recv_g, send_h])], dim=1)
        shifts = torch.cat([sh_hh.repeat(nb, 1), zeros_gg, shift_gh, -shift_gh])
        unit_shifts = torch.cat([us_hh.repeat(nb, 1), zeros_gg, ushift_gh, -ushift_gh])
        assert sh_hh.shape[0] == n_hh

        d = dict(self._template(nb))
        pos = torch.cat([self.host_pos_t[None].expand(nb, -1, -1), g], dim=1).reshape(-1, 3)
        d.update(positions=pos, edge_index=edge_index, shifts=shifts, unit_shifts=unit_shifts)
        out = self._forward(d)
        if self.use_node_energies:
            node = out["node_energy"].double().reshape(nb, n)
            return (node - self.ref_nodes[None]).sum(1).cpu().numpy()
        return (out["energy"].double() - self.E_host - self.E_guest).cpu().numpy()
