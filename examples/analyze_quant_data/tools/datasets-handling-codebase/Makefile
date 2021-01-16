SCRIPT_INTERPETER = python3
SCRIPT_EVAL = create_models_dataset.py
SCRIPT_EVAL_ARGS = --conf_file conf_evaluate_models.yaml
SCRIPT_SUMMARY_EVAL_ARGS = --conf_file conf_evaluate_models.yaml --summary_estimated_workload

create_models_dataset:
	clear
	$(SCRIPT_INTERPETER) $(SCRIPT_EVAL) $(SCRIPT_EVAL_ARGS)

show_summary_workload_create_models_dataset:
	clear
	$(SCRIPT_INTERPETER) $(SCRIPT_EVAL) $(SCRIPT_SUMMARY_EVAL_ARGS)
