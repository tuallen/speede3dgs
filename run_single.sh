#!/usr/bin/env bash
set -e

# List of datasets
datasets=(NeRF-DS D-NeRF HyperNeRF)

# Map each dataset to its scenes
declare -A scenes
scenes["NeRF-DS"]="as_novel_view basin_novel_view bell_novel_view cup_novel_view plate_novel_view press_novel_view sieve_novel_view"
scenes["D-NeRF"]="bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex"
scenes["HyperNeRF"]="broom2 vrig-3dprinter vrig-chicken vrig-peel-banana"

### You can choose one scene mannually from the upper list, we provide some samples below:
dataset=D-NeRF
scene=hook

# dataset=NeRF-DS
# scene=plate_novel_view

# dataset=HyperNeRF/vrig
# scene=broom2

source_path="data/${dataset}/${scene}"
output_path="output/${dataset}/${scene}"

if [ "$dataset" = "D-NeRF" ]; then
  echo ">>> Dataset: $dataset | Scene: $scene"
  python train.py \
    -s "$source_path" \
    -m "$output_path" \
    --eval \
    --is_blender \
    --iterations 30000 \
    --test_iterations 40000 \
    --save_iterations 30000 \
    --gflow_flag
else
  echo ">>> Dataset: $dataset | Scene: $scene"
  python train.py \
    -s "$source_path" \
    -m "$output_path" \
    --eval \
    --iterations 30000 \
    --test_iterations 40000 \
    --save_iterations 30000 \
    --use_apt_noise \
    --gflow_flag
fi


python render.py -m "$output_path" --mode render --gflow_flag --skip_train
python metrics.py -m "$output_path"