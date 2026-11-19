# auto-extracted configuration block of make_vext_nequip.py (MODELS, paths); do not edit here
"""CO2 V_ext in Mg-MOF-74 from published NequIP potentials (cDFT only), 298 K (2026-11-18).

usage: python make_vext_nequip.py <model key> [spacing_A=0.4] [tag=prod]
  model keys: goeminne_pbed3, goeminne_pbets, goeminne_pbembd (Goeminne et al., JCTC 19, 6313 (2023), Zenodo 7904959;
              framework = their rigid 54-atom cell, taken from the first frame of their training data),
              deepmdmc_<structure> (Tayfuroglu & Keskin, JCP 163, 054704 (2025), github otayfuroglu/deepMDMC).

Protocol (same as the uMOF-MH production V_ext, make_production_vext.py, except where noted):
  * grid: cell-centred, `spacing` (0.4 A); no symmetry reduction for the Goeminne framework (P1 within 0.3 A),
    R-3 reduction for the uMOF-MH structure (centred on an inversion centre, as for the uMOF-MH V_ext).
  * host: 2x2x2 supercell for Goeminne (guest-image effects < 0.02 kJ/mol vs 3x3x3, widths 13.2 A); 2x1x1 for
    deepMDMC on uMOF-MH (< 0.004 kJ/mol vs 4x2x2).
  * guest: rigid linear CO2, C at the grid point; C-O = 1.160643 A (Goeminne's co2.xyz) for Goeminne, 1.167767 A
    (uMOF-MH production V_ext) for deepMDMC on uMOF-MH.
  * orientations: 48 linear directions everywhere, 192 on points with best U < -0.30 eV.
  * overlap: computed with a minimal guard only (guest atom - host atom >= 1.0 A, U >= -1.5 eV); the paper's own
    pairwise vdW criterion and our uniform 1.8 A guard are applied afterwards (apply_pair_guard) and saved as
    separate files, so the overlap rule is a reported sensitivity, not a hidden choice.
Writes vext_<key>_<tag>_{raw,paper,u1.8}.npz and a json summary with K_H for every guard.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read

sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.cdft import henry_from_vext  # noqa: E402
from feos_mlip.engine_nequip import NequIPInsertionEngine  # noqa: E402
from feos_mlip.vext import (apply_pair_guard, center_on_inversion, compute_vext, n_points_for_spacing,  # noqa: E402
                            orientations, pair_thresholds_vdw, refine_orientations)

EXT = Path("/local/home/dvoderhol/umof_models_ext")
G = EXT / "goeminne2023"
GSTRUCT = G / "structures/MgMOF74_goeminne_PBED3_framework.cif"
MHSTRUCT = Path("/local/home/dvoderhol/umof_lammps/relaxed_cifs/MgMOF74_prim_uMOF-MH.cif")
GVDW = dict(delta=0.35, overrides={"Mg": 1.0})          # Goeminne gcmc_scripts/run.py + gcmc.py
DVDW = dict(delta=0.2, overrides={"Mg": 1.25})          # deepMDMC runDeepMDMC.py + deep_mdmc.py
# key: (model, structure, guest types, paper overlap rule, host supercell, symmetry reduction, C-O / A)
MODELS = {
    "goeminne_pbed3": (G / "data/training/model_MgMOF74_pbed3.pth", GSTRUCT, ["Co", "Os", "Os"], GVDW, (2, 2, 2), False,
                       1.160642995),
    "goeminne_pbets": (G / "data/training/model_MgMOF74_pbets.pth", GSTRUCT, ["Co", "Os", "Os"], GVDW, (2, 2, 2), False,
                       1.160642995),
    "goeminne_pbembd": (G / "data/training/model_MgMOF74_pbembd.pth", GSTRUCT, ["Co", "Os", "Os"], GVDW, (2, 2, 2),
                        False, 1.160642995),
    # deepMDMC on OUR uMOF-MH relaxed primitive cell (user's choice 2026-11-19: their rigid-GCMC structure is not
    # published); host 2x1x1 as for uMOF-MH (vs 4x2x2: <= 0.004 kJ/mol); C-O as in the uMOF-MH production V_ext.
    "deepmdmc_umofmh": (EXT / "deepmdmc/MgMOF74_CO2.pth", MHSTRUCT, ["C", "O", "O"], DVDW, (2, 1, 1), True,
                        1.1677672052747283),
}
T = 298.0

