
# ============================ #
# Makefile Variables
# ============================ #

BASH_INTERPRETER = bash

FILTER_BASH_SCRIPT = filter_distiller_log_file.sh
FILTER_BASH_SCRIPT_PSNR = filter_log_psnr.sh
FILTER_BASH_SCRIPT_SHOW_PRUNE_TREND = filter_distiller_log_file_show_pruning_trend.sh
FILTER_BASH_SCRIPT_SHOW_BPP = filter_distiller_log_file_show_bpp_trend.sh

FILTER_BASH_SCRIPT_SHOW_BOTH = filter_distiller_log_file_show_overall.sh

FILTER_BASH_SCRIPT_SHOW_BPP_vs_PSNR = filter_distiller_log_file_show_bpp_vs_psnr_trend.sh

BASH_SCRIPT = filter_distiller_log_file_tb.sh


PY_INTERPRETER = python3
PY_SCRIPT = graphis_on_shell.py

TARGET_DIR = '../../siren-project/results/cameramen/distiller-siren/agp_prune'
TARGET_DATE = '___2020.12.14-174343'

DATA_PATH = data.txt
DATA_SPARSITY_PATH = data_sparsity.txt 

DIR_DEST_OUTPUT = '/content/log_analyses'

ARGS = $(TARGET_DIR) $(TARGET_DATE) ${DIR_DEST_OUTPUT}
ARGS_2 = $(TARGET_DIR) $(TARGET_DATE) ${DIR_DEST_OUTPUT} ${DATA_PATH} ${DATA_SPARSITY_PATH}


# ============================ #
# Makefile Tasks
# ============================ #
show_stats_psnr:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT) $(ARGS)

show_stats_psnr_v2:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT_PSNR)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT_PSNR) $(ARGS_2)

show_stats_prune_trend:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT_SHOW_PRUNE_TREND)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT_SHOW_PRUNE_TREND) $(ARGS)

show_overall_stats:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT_SHOW_BOTH)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT_SHOW_BOTH) $(ARGS)

show_stats_bpp:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT_SHOW_BPP)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT_SHOW_BPP) $(ARGS)

show_psnr_vs_bpp:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT_SHOW_BPP_vs_PSNR)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT_SHOW_BPP_vs_PSNR) $(ARGS)


show_tb_stats_curr_run:
	clear
	chmod u+x $(BASH_SCRIPT)
	$(BASH_INTERPRETER) $(BASH_SCRIPT) $(ARGS)
