#!/usr/bin/env bash

clear

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

date_dir="___2020.12.10-044629"
file_name="../../siren-project/results/cameramen/distiller-siren/agp_prune/${date_dir}/${date_dir}.log"


save_data_as_txt_file ${file_name}

# print_raw_table ${file_name}

echo "==== Plot Psnr Trend ===="
experiment_id="events.out.tfevents.1607571992.iside"
experiment_id_path="../../siren-project/results/cameramen/distiller-siren/agp_prune/${date_dir}/${experiment_id}"
# python3 graphis_on_shell.py --input_file data.txt --show_stats_transposed --show_data_from_tb_log
python3 graphis_on_shell.py --input_file data.txt --show_data_from_tb_log --experiment_id "${experiment_id_path}"

exit 0
