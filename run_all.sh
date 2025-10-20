#!/usr/bin/env bash
set -e

# List of datasets
datasets=(NeRF-DS D-NeRF HyperNeRF)

# Map each dataset to its scenes
declare -A scenes
scenes["NeRF-DS"]="as basin bell cup plate press sieve"
scenes["D-NeRF"]="bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex"
scenes["HyperNeRF"]="broom2 vrig-3dprinter vrig-chicken vrig-peel-banana"

# Iterate
for dataset in "${datasets[@]}"; do
  for scene in ${scenes[$dataset]}; do
    source_path="data/${dataset}/${scene}"
    for v in 1 2 3; do
      output="output/${dataset}/${scene}_v${v}"
      pc="$output/point_cloud/iteration_30000/point_cloud.ply"
      fps="$output/fps_30000.txt"
      numg="$output/num_gaussians_30000.txt"
      results="$output/results.json"

      echo ">>> Dataset: $dataset | Scene: $scene | Version: v${v}"

      # if no point cloud, (re)train
      if [ ! -f "$pc" ]; then
        rm -rfv "$output"

        if [ "$dataset" = "D-NeRF" ]; then
          # echo "D-NeRF, use_apt_noise=False"
          python train.py \
            -s "$source_path" \
            -m "$output" \
            --eval \
            --iterations 30000 \
            --test_iterations 40000 \
            --save_iterations 30000 \
            --gflow_flag
        else
          # echo "not D-NeRF, use_apt_noise=True"
          python train.py \
            -s "$source_path" \
            -m "$output" \
            --eval \
            --iterations 30000 \
            --test_iterations 40000 \
            --save_iterations 30000 \
            --use_asp \
            --gflow_flag
        fi

        # if still missing, clean up
        if [ ! -f "$pc" ]; then
          rm -rfv "$output"
        fi
      fi
    done
  done
done