#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python3 -u ../src/train.py \
#   --config_path ../config/config_llama_chat.yaml \
#   --dataset_name WaterDrum-TOFU \
#   --subset forget_10 \
#   --split forget \
#   --seed 42

CUDA_VISIBLE_DEVICES=3 python3 -u ../src/train.py \
  --config_path ../config/config_llama.yaml \
  --dataset_name WaterDrum-Ax \
  --subset forget_05 \
  --split forget \
  --seed 42