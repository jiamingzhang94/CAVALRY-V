#!/bin/bash

set -euo pipefail

export HUGGINGFACE_HUB_TOKEN=${YOUR_HUGGINGFACE_HUB_TOKEN}
export LMUData=${YOUR_DATA_PATH}

datasets=(
#   "MMBench_Video_64frame_nopack"
  "MMBench_Video_64frame_nopack_cavalry"

#   "MME"
#   "MME_cavalry"
)
# models=(
# "Qwen2.5-VL-7B-Instruct"
# "llava_video_qwen2_7b"
# "Aria"
# "InternVL2_5-8B"
# "MiniCPM-o-2_6"
# )

for data in "${datasets[@]}"; do
  echo "==== Running on dataset: $data ===="
  torchrun --nproc-per-node=8 run.py \
    --model llava_video_qwen2_7b \
    --data "$data" \
    --work-dir "output_${data}" \
    # --judge-args '{"use_azure": true}' \
done