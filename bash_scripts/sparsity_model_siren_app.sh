#!/usr/bin/env bash

clear

# =================================================================== #
# Global Variables
# =================================================================== #

SCRIPT_SPARSITY_CHECK=siren_main_app.py
SCRIPT_INTERPETER=python3

declare -A SPARSITY_DEFAULT_OPTS=\
( ["LOGGING_ROOT_SUMMARY"]="../sparsity_summaries" \
 ["SIDELENGHT"]=256 \
 ["N_HF"]=64 \
 ["N_HL"]=5)

declare -A SPARSITY_OPTS=\
( ["LOGGING_ROOT_SUMMARY"]="../sparsity_summaries" \
 ["SIDELENGHT"]=256 \
 ["N_HF"]=64 \
 ["N_HL"]=5)

# =================================================================== #
# Functions
# =================================================================== #


# ------------------------------------------------------------------- #
# Utils
# ------------------------------------------------------------------- #
function log_debug() {
  declare msg=$1
  echo "[DEBUG] $msg"
}

function print_info() {
  declare msg=$1
  echo "[*] $msg"
}

function check_file_exists() {
  declare a_file=$1
  if [ ! -f $a_file ] ; then
    echo "Error:'$a_file' does not exists!"
    exit -1
  fi
}

function print_hash_map_arr() {
  # Print Hash-Mpa Array key-value pairs.
  declare -n __p="$1"
  for k in "${!__p[@]}" ; do
     printf "%s=%s\n" "$k" "${__p[$k]}" ;
  done ;
}

function read_file() {
  declare a_file=$1
  while IFS= read -r line ; do
    echo "$line"
  done < $a_file
}

function print_arr() {
  declare a_arr=$@
  echo "Array's length: $#"
  for i in $a_arr ; do
    echo -e "$i"
  done
}

# ------------------------------------------------------------------- #
# Sparsity Handling Functions
# ------------------------------------------------------------------- #
function update_sparsity_opts_map() {
  declare conf_arr=$@
  # log_debug "Inside update_sparsity_opts_map()"
  for i in $conf_arr ; do
    # log_debug $i
    a_key_opt=$(echo $i | cut -d "=" -f 1)
    a_value_opt=$(echo $i | cut -d "=" -f 2)
    # log_debug "key=$a_key_opt val=$a_value_opt"
    SPARSITY_OPTS[$a_key_opt]=$a_value_opt
  done
}

function setup_sparsity_opts() {
  declare conf_file=$1
  # print_info "Show un-processed input file:"
  # read_file $conf_file
  
  conf_file_cleaned=$(cat $conf_file | grep -v "^#" | grep -v "\s+")
  conf_arr=$(echo $conf_file_cleaned | tr "" "\n")
  # print_info "Show array of options:"
  # print_arr $conf_arr
  
  # print_info "Sparsity Options Before:"
  # print_hash_map_arr SPARSITY_OPTS
  
  update_sparsity_opts_map $conf_arr
  
  # print_info "Sparsity Options After:"
  # print_hash_map_arr SPARSITY_OPTS
}

function run_sparsity_check() {
  # log_debug "Inside run_sparsity_check() "
  declare -n sparsity_options="$1"
  # print_hash_map_arr sparsity_options
  
  LOGGING_ROOT_SUMMARY="${sparsity_options[LOGGING_ROOT_SUMMARY]}"
  SIDELENGHT="${sparsity_options[SIDELENGHT]}"
  N_HF="${sparsity_options[N_HF]}"
  N_HL="${sparsity_options[N_HL]}"
  STATE_DICT_MODEL_FILE="${sparsity_options[STATE_DICT_MODEL_FILE]}"
  
  # log_debug "Check if exists file '${STATE_DICT_MODEL_FILE}'..."
  check_file_exists $STATE_DICT_MODEL_FILE
  # log_debug "It exists!"
  # exit 0
  
  $SCRIPT_INTERPETER $SCRIPT_SPARSITY_CHECK \
    --logging_root ${LOGGING_ROOT_SUMMARY} \
    --experiment_name 'train' \
    --sidelength ${SIDELENGHT} \
    --n_hf ${N_HF} \
    --n_hl ${N_HL} \
    --cuda \
    --evaluate \
    --verbose 0 \
    --cuda \
    --summary 'sparsity' \
    --save-image-on-test \
    --exp-load-weights-from ${STATE_DICT_MODEL_FILE}
}

function check_cmd_line_args() {
  # Check command line args
  # log_debug "Inside check_cmd_line_args()"
  if [ $# -eq 1 ] ; then
    # log_debug "Conf file detected, checking it..."
    conf_file=$1
    setup_sparsity_opts $conf_file
    # print_hash_map_arr SPARSITY_OPTS
  fi
}

# =================================================================== #
# Script starts executing from here
# =================================================================== #

# Check command line args
check_cmd_line_args $@
# print_hash_map_arr SPARSITY_OPTS

run_sparsity_check SPARSITY_OPTS

exit 0

