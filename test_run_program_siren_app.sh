#!/usr/bin/env bash

# =================================================== #
# Script: siren_main_app.sh
#
# - Used it for launching a run for test
#   a deep learning model based on Siren-like
#   Architecture for retrieving model's performance
#   metrices that are related to Cameramen compressed
#   image.
# =================================================== #

SCRIPT_SIREN_APP=siren_main_app.py
SCRIPT_INTERPRETER=python3

# LOGGING_ROOT='../../../results/cameramen/distiller-siren'
LOGGING_ROOT='../results/tests'

STATE_DICT_MODEL_FILE="/media/franec94/OS/data/data_thesys/distiller-siren/cameramen/agp_pruning/1st_battery/___2020.12.06-201125/_final_ckpt_epoch_449999.pth.tar"

CUDA_VISIBLE_DEVICES=0 \
  $SCRIPT_INTERPRETER $SCRIPT_SIREN_APP \
    --logging_root $LOGGING_ROOT \
    --experiment_name 'test_model' \
    --sidelength 256 \
    --n_hf 64  \
    --n_hl 5 \
    --seed 0 \
    --cuda \
    --evaluate \
    --exp-load-weights-from ${STATE_DICT_MODEL_FILE} \
    --verbose 0 \

exit 0
