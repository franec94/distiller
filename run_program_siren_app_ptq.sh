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
    --logging_root '../../../results/cameramen/distiller-siren/evals/ptq-eval' \
    --experiment_name 'test' \
    --sidelength {SIDELENGHT} \
    --n_hf 64 \
    --n_hl 5 \
    --seed 0 \
    --cuda \
    --evaluate \
    --verbose 0 \
    --save-image-on-test \
    --exp-load-weights-from ""\
    --quantize-eval \
    --qe-config-file "" \
    --qe_lapq
