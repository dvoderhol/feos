#!/bin/bash
#SBATCH -J umof-isochk
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o isochk_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_ch4
python isotherm_bug_check.py
