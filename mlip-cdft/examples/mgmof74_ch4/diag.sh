#!/bin/bash
#SBATCH -J umof-diag
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o diag_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_ch4
python diagnose_highp.py
