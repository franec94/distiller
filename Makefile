# =============================================== #
# Makefile - Variables:
# =============================================== #

# Plain Train scripts
# --------------------------
RUN_PROGRAM = run_program.sh
RUN_PROGRAM_APP = run_program_siren_app.sh
RUN_PROGRAM = run_program_siren_app.sh

# Test scripts
# --------------------------
TEST_RUN_PROGRAM = test_run_program_siren_app.sh

# Prunining scripts
# --------------------------
RUN_PROGRAM_AGP_PRUNE = run_program_siren_app_pruning_agp.sh


# =============================================== #
# Makefile - Targets:
# =============================================== #

train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM)
	./$(RUN_PROGRAM)
siren_app_train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM_APP)
	./$(RUN_PROGRAM_APP)

siren_app_train_on_cameramen_agp_prune:
	clear
	chmod u+x $(RUN_PROGRAM_AGP_PRUNE)
	./$(RUN_PROGRAM_AGP_PRUNE)

test_siren_app_train_on_cameramen:
	clear
	chmod u+x $(TEST_RUN_PROGRAM)
	./$(TEST_RUN_PROGRAM)

install_distiller_dependencies_via_requirements_file:
	clear
	pip install -r requirements.txt
