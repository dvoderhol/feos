#!/bin/bash
#SBATCH -J umof-adapt
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o adapt_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_ch4
python adaptive_test.py
