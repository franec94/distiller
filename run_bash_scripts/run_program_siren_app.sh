#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# =============================================== #

CUDA_VISIBLE_DEVICES=0 python main.py \
  --logging_root '../../../results/cameramen/distiller-siren/' \
  --experiment_name 'train' \
  --sidelength 256 \
  --num_epochs 500000 \
  --n_hf 64  \
  --n_hl 5 \
  --lambda_L_1 0 \
  --lambda_L_2 0 \
  --num-best-scores 3 \
  --epochs_til_ckpt 5000 \
  --save_mid_ckpts 99999 149999 199999 249999 299999 349999 399999 449999 499999  \
  --seed 0 \
  --cuda \
  --train \
  --evaluate \
  --verbose 0

exit 0
