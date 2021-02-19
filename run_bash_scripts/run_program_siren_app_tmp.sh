#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# =============================================== #

CUDA_VISIBLE_DEVICES=0 python siren_main_app.py \
  --logging_root '../../../results/cameramen/distiller-siren/fine_tune' \
  --experiment_name 'train' \
  --compress "../../../schedulers/fine-tuning/siren64_5.schedule_fine_tuning.yaml" \
  --sidelength 256 \
  --n_hf 64  \
  --n_hl 5 \
  --seed 0 \
  --cuda \
  --num_epochs 550000 \
  --lr 0.0001 \
  --lambda_L_1 0 \
  --lambda_L_2 0 \
  --epochs_til_ckpt 750 \
  --num-best-scores 5 \
  --train \
  --evaluate \
  --verbose 0 \
  --resume-from "../../../ckpts/_final_ckpt_epoch_449999.pth.tar"

#  --resume-from "../../../ckpts/_mid_ckpt_epoch_299999.pth.tar"

# --compress ./examples/agp-pruning/siren64_5.schedule_agp.yaml \
# --save_mid_ckpts 99999 149999 174999 199999 \
exit 0
