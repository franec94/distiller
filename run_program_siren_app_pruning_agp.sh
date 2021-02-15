#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# =============================================== #

IMAGE_FILEPATH="../../../BSD68/test066.png"

LOGGING_ROOT="../../../results/test066/distiller-siren/agp_prune"

SCHEDULER_FILEPATH="../../../schedulers/test066_airplane/agp-pruning/siren64_5.schedule_agp.yaml"

INIT_FROM="../../../ckpts/test066_airplane/_mid_ckpt_epoch_299999.pth.tar"

CUDA_VISIBLE_DEVICES=0 python3 siren_main_app.py \
  --logging_root ${LOGGING_ROOT} \
  --experiment_name 'train' \
  --compress  ${SCHEDULER_FILEPATH} \
  --image_filepath ${IMAGE_FILEPATH} \
  --sidelength 256 \
  --n_hf 64  \
  --n_hl 5 \
  --seed 0 \
  --cuda \
  --num_epochs 475000 \
  --lr 0.0001 \
  --lambda_L_1 0 \
  --lambda_L_2 0 \
  --epochs_til_ckpt 900 \
  --num-best-scores 3 \
  --train \
  --evaluate \
  --verbose 0 \
  --resume-from  ${INIT_FROM} \
  --target_sparsity 30.0 \
  --toll_sparsity 2.0 \
  --patience_sparsity 1000 \
  --trail_epochs 1000 \
  --mid_target_sparsities 5 10 20 25 30 35 40

# --compress ./examples/agp-pruning/siren64_5.schedule_agp.yaml \
# --save_mid_ckpts 99999 149999 174999 199999 \
exit 0
