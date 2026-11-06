#!/bin/bash
#SBATCH -J umof-co2
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o stage2_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_co2
python stage2_isotherms.py "$@"
