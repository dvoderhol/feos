#!/bin/bash
#SBATCH -J br-uff
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o branches_uff_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_co2
python branches_m1.py uff branches_uff.json
