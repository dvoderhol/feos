#!/bin/bash
#SBATCH -J co2final
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o final_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_co2
python final_isotherms.py
