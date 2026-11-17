#!/bin/bash
#SBATCH -J followup
#SBATCH -p LocalQ
#SBATCH -c 12
#SBATCH -o followup_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=12
cd ~/feos/mlip-cdft/examples/mgmof74_co2
python review_followup.py
