#!/bin/bash
# V_ext for the three Goeminne 2023 NequIP models, one after another on GPU $1 (checked free before launch)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate umof
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/feos/mlip-cdft/examples/mgmof74_co2/ext_models
for k in goeminne_pbembd goeminne_pbed3 goeminne_pbets; do
  python make_vext_nequip.py $k 0.4 prod > log_vext_${k}.txt 2>&1 || echo "FAILED $k" >> log_vext_${k}.txt
done
echo done > vext_goeminne.done
