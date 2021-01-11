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

clear

function check_file_exists() {
    local file_path=$1
    # echo -e "[*] check file - '${file_path}' exists..."
    if [ ! -f "${file_path}" ] ; then
        echo "Error: '${file_path}' is not a file."
        exit -1
    fi
    # echo -e "[*] check file - Done."
}

printarr() { declare -n __p="$1"; for k in "${!__p[@]}"; do printf "%s=%s\n" "$k" "${__p[$k]}" ; done ;  } 

function run_trials_linear_quant() {
    local INITIALIZED_MODEL=$1
    local RESULTS_CSV_PATH=$2

    LOGGING_ROOT='../../../results/cameramen/distiller-siren/train/quant-aware/linear-quant/middle-freq/attempt_1'
    COMPRESS_SCHEDULE="../../../schedulers/quant-aware-training/siren_quant_aware_train_linear_quant.yaml"
    COMPRESS_COMBS="../../../schedulers/quant-aware-training/siren_quant_aware_train_linear_quant_long_train.csv"

    check_file_exists $COMPRESS_SCHEDULE
    check_file_exists $COMPRESS_COMBS

    declare -A MAP_OPTS
    declare -A MAP_IDX_OPT
    i=0
    while IFS= read -r line ; do
        # echo "$line"
        line=$(echo -n "$line" | xargs)

        if [ "$line" == "" ] ; then
            echo "Empty Line Skipped!"
            continue
        fi

        options=$(echo $line | tr -d '\r' | tr "," " ")
        if [ $i -eq 0 ] ; then
            j=0 
            for opt in $options ; do
                # echo "> [$opt]"
                MAP_OPTS[${opt}]=""
                MAP_IDX_OPT[$j]=$opt
                # echo "${MAP_IDX_OPT[$j]}"
                j=$((j+1))
            done
	# printarr MAP_IDX_OPT
        else
            j=0 
            for opt in $options ; do
                # echo "> [$opt]"
                opt_name="${MAP_IDX_OPT[${j}]}"
                MAP_OPTS["$opt_name"]=${opt}
                # echo "$opt_name --> ${MAP_OPTS[$opt_name]}"
                j=$((j+1))
            done
	    printarr MAP_OPTS
            pos_comb=$((i-1))
            python3 update_quant_scheduler.py \
                --compress $COMPRESS_SCHEDULE \
                --combs $COMPRESS_COMBS \
                --pos_comb $pos_comb
            run_trials $LOGGING_ROOT $COMPRESS_SCHEDULE $INITIALIZED_MODEL ${MAP_OPTS[epochs]} ${MAP_OPTS[lr]} $RESULTS_CSV_PATH
            # clear
            # echo -- ${MAP_OPTS[epochs]}
            # echo -- ${MAP_OPTS[lr]}
            # exit 0
        fi
        i=$((i+1))
    done < "$COMPRESS_COMBS"
}

function run_trials() {
    local LOGGING_ROOT=$1
    local COMPRESS_SCHEDULE=$2
    local INITIALIZED_MODEL=$3
    local EPOCHS=$4
    local LR=$5
    local RESULTS_CSV_PATH=$6
    # echo LR $LR EPOCHS $EPOCHS
    # return
    CUDA_VISIBLE_DEVICES=0 python3 siren_main_app.py \
        --logging_root ${LOGGING_ROOT} \
        --experiment_name 'quant_aware_train' \
        --sidelength 256 \
        --n_hf 64 \
        --n_hl 5 \
        --seed 0 \
        --cuda \
        --train \
        --evaluate \
        --num_epochs ${EPOCHS} \
        --lr ${LR} \
        --epochs_til_ckpt 5000 \
        --verbose 0 \
        --resume-from ${INITIALIZED_MODEL} \
        --compress ${COMPRESS_SCHEDULE} \
        --save_test_data_to_csv_path ${RESULTS_CSV_PATH}
}

INITIALIZED_MODEL="../../../ckpts/_final_ckpt_epoch_430729.pth.tar"
RESULTS_CSV_PATH="../../../results/csv/quant-aware-train/linear-quant/linear_quant_attempt_1.csv"

run_trials_linear_quant $INITIALIZED_MODEL $RESULTS_CSV_PATH

exit 0


LOGGING_ROOT='../../../results/cameramen/distiller-siren/train/quant-aware/dorefa-quant'
COMPRESS_SCHEDULE="../../../quant-configs/siren_quant_aware_train_linear_quant.yaml"

LOGGING_ROOT='../../../results/cameramen/distiller-siren/train/quant-aware/wrpn-quant'
COMPRESS_SCHEDULE="../../../quant-configs/siren_quant_aware_train_linear_quant.yaml"

LOGGING_ROOT='../../../results/cameramen/distiller-siren/train/quant-aware/pact-quant'
COMPRESS_SCHEDULE="../../../quant-configs/siren_quant_aware_train_linear_quant.yaml"

# Option: "exp-load-weights-from":
# (1) --exp-load-weights-from "../../../ckpts/_mid_ckpt_epoch_299999.pth.tar" \
# (2) --exp-load-weights-from "../../../ckpts/_final_ckpt_epoch_449999.pth.tar" \

exit 0
