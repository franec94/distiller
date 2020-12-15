#!/usr/bin/env bash

clear
# ------------------------------------------------ #
# Functions Section
# ------------------------------------------------ #

# src_data="/content/data.txt"
# src_sparsity_data="/content/data_sparsity.txt"


function check_out_cmd_args() {
    script_basename=$(echo $0 | basename)
    if [ $# -ne 5 ] ; then
      script_basename
      printf "Usage: ${script_basename} {TARGET_DIR} {TARGET_DATE} {DIR_DEST_OUTPUT} {DATA_PATH} {SPARSITY_DATA_PATH}"
      exit -1
    fi
    date_dir=$2
    file_name_1="$1/${date_dir}/${date_dir}.log"
    if [ ! -f "${file_name_1}" ] ; then 
      file_name_1="$1/${date_dir}.log"
      if [ ! -f "${file_name_2}" ] ; then 
        printf "Both ${file_name_1} and ${file_name_2} do not exist!"
        exit -2
      fi
    fi   
}

check_out_cmd_args $@

src_data=$4
src_sparsity_data=$5


function save_data_as_txt_file() {
  local file_name=$1
  cat ${file_name} \
    | grep -v -e "^.*Pruner.*$" \
    | grep -e "^.*MSE.*$" \
    | grep -n -v "^.*Best.*$" \
    | cut -d " " -f 10 \
    > ${src_data}

  unset file_name

}

function save_sparsity_data_as_txt_file() {
  local file_name=$1

  cat ${file_name} \
    | grep -e "^.*Total sparsity.*$" \
    | cut -d " " -f 6 \
    | grep -v "sparsity" \
    > ${src_sparsity_data}

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
  last_recorded=$(cat ${src_data} | tail -n 1)
  last_epoch=$(cat ${file_name} \
    | grep -e "^.*epoch=.*$" \
    | tail -n 1 )

  echo "Last entry: ${last_recorded}"
  echo "Last epoch: ${last_epoch}"
}

function get_avg_via_awk() {
  # cat data.txt | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1); tot_lines++; printf("%.2f\n", $1)} END{printf("Avg:\t\t%.2f\n", acc/tot_lines)} '
  cat ${src_data} | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1); tot_lines++;} END{printf("Avg: %.2f\n", acc/tot_lines)} '
}

function get_std_via_awk() {
  avg_data=$(get_avg_via_awk | cut -d ":" -f 2)
  cat ${src_data} | awk -v mean="${avg_data}" 'BEGIN{tot_lines=0; acc=0;} {acc+=($1-mean)*($1-mean); tot_lines++;} END{printf("Std: +/-%.2f\n", sqrt(acc/tot_lines))} '
}

function get_var_via_awk() {
  cat ${src_data} | awk 'BEGIN{tot_lines=0; acc=0;} {acc+=($1-mean)*($1-mean); tot_lines++;} END{printf("Var: %.2f\n", acc/tot_lines)} '
}

function show_curr_epoch() {
  echo "==== Show Epochs Achieved ===="
  curr_epoch=$(cat ${file_name} | grep -e "^.*epoch=.*$" | tail -n 1 |cut -d '=' -f 2 | cut -d "(" -f 1 | cut -d ')' -f 1)
  target_epochs=$(cat ${file_name} | grep -e "^.*--num_epochs.*$" | tail -n 1 \
    | awk --field-separator="--" '{for(i=0; i < NF; i++) { if($i ~ /^num_epochs/) {printf("%s\n", $i); break;}}}'\
    | cut -d " " -f 2)
  echo "[*] Epoch: $curr_epoch / $target_epochs"
  remaining_epochs=$( echo ${target_epochs} - ${curr_epoch} | bc -l )
  echo "[*] Still to go: $remaining_epochs epochs"
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
  echo -e "[*] Still to go: ${remaining_sparsity} sparsity"
}

# ------------------------------------------------ #
# Script Starts Here
# ------------------------------------------------ #

# Data Infos.
date_dir=$2
file_name="$1/${date_dir}/${date_dir}.log"
if [ ! -f "${file_name}" ] ; then
  file_name="$1/${date_dir}.log"
fi
output_dir=$3

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
cmd_as_str="python3 graphics_on_shell.py \n --output_dir ${output_dir} \n --input_file ${src_data} \n --input_file_pruning_trend ${src_sparsity_data}  \n --show_graphics \n --show_psnr_trend \n --show_stats_transposed --show_data_from_log"
echo -e ${cmd_as_str}

python3 graphics_on_shell.py \
  --output_dir ${output_dir} \
  --input_file ${src_data} \
  --input_file_pruning_trend ${src_sparsity_data}  \
  --show_graphics \
  --show_psnr_trend \
  --show_stats_transposed --show_data_from_log
if [ $? -ne 0 ] ; then
  printf "Program ended wrongly with exit code: %d\n" $?
else
  printf "Program ended correctly with exit code: %d\n" $?
fi

exit 0