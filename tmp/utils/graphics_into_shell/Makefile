
# ============================ #
# Makefile Variables
# ============================ #

TARGET_DIR = '../../siren-project/results/cameramen/distiller-siren/agp_prune'
TARGET_DATE = '___2020.12.14-174343'
DIR_DEST_OUTPUT = '/content/log_analyses'

ARGS = $(TARGET_DIR) $(TARGET_DATE) ${DIR_DEST_OUTPUT}

FILTER_BASH_SCRIPT = filter_distiller_log_file.sh
FILTER_BASH_SCRIPT_SHOW_PRUNE_TREND = filter_distiller_log_file_show_pruning_trend.sh
FILTER_BASH_SCRIPT_SHOW_BOTH = filter_distiller_log_file_show_overall.sh
FILTER_BASH_SCRIPT_SHOW_BPP = filter_distiller_log_file_show_bpp_trend.sh
FILTER_BASH_SCRIPT_SHOW_BPP_vs_PSNR = filter_distiller_log_file_show_bpp_vs_psnr_trend.sh
BASH_SCRIPT = filter_distiller_log_file_tb.sh
BASH_INTERPRETER = bash


PY_SCRIPT = graphis_on_shell.py
PY_INTERPRETER = python3

# ============================ #
# Makefile Tasks
# ============================ #
show_stats_psnr:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT) $(ARGS)

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
