#!/bin/bash
#SBATCH -J umof-rerun
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o rerun_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd ~/feos/mlip-cdft/examples/mgmof74_co2 && python stage2_isotherms.py vext_CO2_MH_0.4A_24.npz 0.3 iso_CO2_298K_v2.json
cd ~/feos/mlip-cdft/examples/mgmof74_ch4 && python adaptive_test.py
