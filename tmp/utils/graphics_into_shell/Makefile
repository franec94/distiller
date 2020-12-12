
# ============================ #
# Makefile Variables
# ============================ #
FILTER_BASH_SCRIPT = filter_distiller_log_file.sh
BASH_SCRIPT = filter_distiller_log_file_tb.sh
BASH_INTERPRETER = bash


PY_SCRIPT = graphis_on_shell.py
PY_INTERPRETER = python3

# ============================ #
# Makefile Tasks
# ============================ #
show_stats_curr_run:
	clear
	chmod u+x $(FILTER_BASH_SCRIPT)
	$(BASH_INTERPRETER) $(FILTER_BASH_SCRIPT)

show_tb_stats_curr_run:
	clear
	chmod u+x $(BASH_SCRIPT)
	$(BASH_INTERPRETER) $(BASH_SCRIPT)
