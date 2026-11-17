#!/bin/bash
#SBATCH -J canon-mlip
#SBATCH -p LocalQ
#SBATCH -c 8
#SBATCH -o canonloop_mlip_%j.out
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export RAYON_NUM_THREADS=8
cd ~/feos/mlip-cdft/examples/mgmof74_co2
python canonical_loop.py mlip
