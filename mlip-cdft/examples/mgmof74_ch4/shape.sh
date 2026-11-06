#!/bin/bash
#SBATCH -J umof-shape
#SBATCH -p LocalQ
#SBATCH -c 12
#SBATCH -o shape_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=12
cd ~/feos/mlip-cdft/examples/mgmof74_ch4
python cell_shape_test.py
