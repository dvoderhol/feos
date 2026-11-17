#!/bin/bash
# uMOF-POLAR CO2 V_ext (same protocol as uMOF-MH) on GPU 0: 0.4 A grid, 48 linear directions, then 192 on deep points + 1.8 A guard
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
cd ~/feos/mlip-cdft/examples/mgmof74_co2
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python stage1_vext.py 0.4 48 vext_CO2_POLAR_0.4A_lin48.npz linear uMOF-POLAR > stage1_CO2_POLAR_lin48.log 2>&1
python make_production_vext_polar.py > make_production_vext_polar.log 2>&1
sbatch final_polar.sh
