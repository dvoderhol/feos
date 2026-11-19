#!/bin/bash
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o ext_iso_%x_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd ~/feos/mlip-cdft/examples/mgmof74_co2/ext_models
python ext_isotherms.py "$@"
