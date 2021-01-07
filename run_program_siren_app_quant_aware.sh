#!/usr/bin/env bash

# ======================================================== #
# Script: siren_main_app.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed image:
# 
# (1.a) By means of just a pre-trained model;
# (1.b) By means of pre-trained and already pruned model.
# (2) Attempting to quantize it, via yaml scheduler.
# ======================================================== #

CUDA_VISIBLE_DEVICES=0 python siren_main_app.py \
    --logging_root '../../../results/cameramen/distiller-siren/train/quant-aware' \
    --experiment_name 'quant_aware_train' \
    --sidelength 256 \
    --n_hf 64 \
    --n_hl 5 \
    --seed 0 \
    --cuda \
    --num_epochs 475000 \
    --lr 0.0001 \
    --verbose 0 \
    --exp-load-weights-from "../../../ckpts/_mid_ckpt_epoch_299999.pth.tar" \
    --compress "../../../quant-configs/siren_quant_aware_train_linear_quant.yaml" \


# Option: "exp-load-weights-from":
# (1) --exp-load-weights-from "../../../ckpts/_mid_ckpt_epoch_299999.pth.tar" \
# (2) --exp-load-weights-from "../../../ckpts/_final_ckpt_epoch_449999.pth.tar" \

exit 0
