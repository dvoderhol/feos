#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/feos/mlip-cdft/examples/mgmof74_co2/ext_models
python widom_check.py deepmdmc_umofmh 500000 > log_widom_deepmdmc.txt 2>&1
python widom_check.py goeminne_pbed3 500000 > log_widom_goeminne_pbed3.txt 2>&1
