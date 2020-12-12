#!/usr/bin/env bash

clear
# ------------------------------------------------ #
# Functions Section
# ------------------------------------------------ #

function save_data_as_txt_file() {
  local file_name=$1
  cat ${file_name} \
    | grep -v -e "^.*Pruner.*$" \
    | grep -e "^.*MSE.*$" \
    | grep -n -v "^.*Best.*$" \
    | cut -d " " -f 10 \
    > data.txt

  unset file_name

}

function save_sparsity_data_as_txt_file() {
  local file_name=$1

  cat ${file_name} \
    | grep -e "^.*Total sparsity.*$" \
    | cut -d " " -f 6 \
    | grep -v "sparsity" \
    > data_sparsity.txt

  unset file_name

}

function print_raw_table() {
  local file_name=$1
  echo "==== Show sorted data ===="
  header="|Occrs.|Data|" \
  line_str="-----------------------" \

  echo ""
  echo  "${line_str}" \
    && echo -e "${header}" \
    && echo  "${line_str}" \
    && cat ${file_name} \
    | grep -v -e "^.*Pruner.*$" \
    | grep -e "^.*MSE.*$" \
    | grep -n -v "^.*Best.*$" \
    | cut -d " " -f 10 | sort | uniq -c \
    | head -n 5
  echo "      ............"
  cat ${file_name} \
    | grep -v -e "^.*Pruner.*$" \
    | grep -e "^.*MSE.*$" \
    | grep -n -v "^.*Best.*$" \
    | cut -d " " -f 10 | sort | uniq -c \
    | tail -n 5
  echo  "${line_str}"
  get_stats
  tot_lines=$(cat data.txt | wc -l)
  echo "Tot Lines.: ${tot_lines}"
  echo  "${line_str}"
  last_entry_recorded
  echo  "${line_str}"

  echo ""
  unset file_name
}


function get_stats() {
  get_avg_via_awk
  get_var_via_awk
  get_std_via_awk
}

function last_entry_recorded() {
  last_recorded=$(cat data.txt | tail -n 1)
  last_epoch=$(cat ${file_name} \
    | grep -e "^.*epoch=.*$" \
    | tail -n 1 )

  echo "Last entry: ${last_recorded}"
  echo "Last epoch: ${last_epoch}"
}

function get_avg_via_awk() {
  # cat data.txt | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1); tot_lines++; printf("%.2f\n", $1)} END{printf("Avg:\t\t%.2f\n", acc/tot_lines)} '
  cat data.txt | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1); tot_lines++;} END{printf("Avg: %.2f\n", acc/tot_lines)} '
}

function get_std_via_awk() {
  avg_data=$(get_avg_via_awk | cut -d ":" -f 2)
  cat data.txt | awk -v mean="${avg_data}" 'BEGIN{tot_lines=0; acc=0;} {acc+=($1-mean)*($1-mean); tot_lines++;} END{printf("Std: +/-%.2f\n", sqrt(acc/tot_lines))} '
}

function get_var_via_awk() {
  cat data.txt | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1-mean)*($1-mean); tot_lines++;} END{printf("Var: %.2f\n", acc/tot_lines)} '
}

function show_curr_epoch() {
  echo "==== Show Epochs Achieved ===="
  curr_epoch=$(cat ${file_name} | grep -e "^.*epoch=.*$" | tail -n 1 |cut -d '=' -f 2 | cut -d "(" -f 1 | cut -d ')' -f 1)
  target_epochs=$(cat ${file_name} | grep -e "^.*--num_epochs.*$" | tail -n 1 \
    | awk --field-separator="--" '{for(i=0; i < NF; i++) { if($i ~ /^num_epochs/) {printf("%s\n", $i); break;}}}'\
    | cut -d " " -f 2)
  echo "[*] $curr_epoch / $target_epochs"
  remaining_epochs=$( echo ${target_epochs} - ${curr_epoch} | bc -l )
  echo "[*] Still to go $remaining_epochs"
}

function show_sparsity_details() {
  echo "==== Show Sparsity Level Achieved ===="

  target_sparsity=$(cat ${file_name} | grep -e "^.*--target_sparsity.*$" | tail -n 1 \
    | awk --field-separator="--" '{for(i=0; i < NF; i++) { if($i ~ /^target_sparsity/) {printf("%s\n", $i); break;}}}')
  #  | awk --field-separator="--" '{for(i=0; i < NF; i++) { if($i ~ /^target_sparsity/) {printf("%s\n", $i); break;} else {printf("%s\n", $i)}}}'
  echo -e "[*] ${target_sparsity}"
  reached_sparsity=$(cat ${file_name} | grep -e "^.*Total sparsity.*$" | grep -v -e "^.*Best.*$" | tail -n 1)
  echo -e "[*] Currently Achieved at ${reached_sparsity}"

  val_t=$(echo ${target_sparsity} | cut -d " " -f 2)
  val_r=$(echo ${reached_sparsity} | cut -d " " -f 6)
  remaining_sparsity=$( echo ${val_t} - ${val_r} | bc -l )
  echo -e "[*] Still to go ${remaining_sparsity}"
}

# ------------------------------------------------ #
# Script Starts Here
# ------------------------------------------------ #

# Data Infos.
date_dir="___2020.12.12-190415"
file_name="../../siren-project/results/cameramen/distiller-siren/agp_prune/${date_dir}/${date_dir}.log"

# Process raw log file.
save_data_as_txt_file ${file_name}
save_sparsity_data_as_txt_file ${file_name}

# print_raw_table ${file_name}

# Show some basic infos.
show_curr_epoch
show_sparsity_details

# Show data extraced from raw log as graphics
# by means of python3 based script.
echo "==== Plot Psnr Trend ===="
# python3 graphis_on_shell.py \
#  --input_file data.txt \
#  --show_stats_transposed --show_data_from_log

python3 graphis_on_shell.py \
  --input_file data.txt \
  --show_both_same_graphics \
  --show_psnr_trend \
  --input_file_pruning_trend data_sparsity.txt  \
  --show_stats_transposed --show_data_from_log

exit 0
