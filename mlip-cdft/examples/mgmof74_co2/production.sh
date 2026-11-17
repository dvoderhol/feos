#!/bin/bash
# production V_ext (GPU 0) then cDFT isotherms (CPU via Slurm)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
cd ~/feos/mlip-cdft/examples/mgmof74_co2
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python make_production_vext.py > make_production_vext.log 2>&1
sbatch stage3.sh vext_CO2_MH_production.npz 0.3 iso_CO2_m1_298K.json
