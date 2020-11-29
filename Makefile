# ======================== #
# Variables
# ======================== #

RUN_PROGRAM = run_program.sh
RUN_PROGRAM_APP = run_program_siren_app.sh
RUN_PROGRAM = run_program_siren_app.sh
TEST_RUN_PROGRAM = test_run_program_siren_app.sh


# ======================== #
# Targets
# ======================== #

run_program_train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM)
	./$(RUN_PROGRAM)

run_program_siren_app_train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM_APP)
	./$(RUN_PROGRAM_APP)

test_run_program_siren_app_train_on_cameramen:
	clear
	chmod u+x $(TEST_RUN_PROGRAM)
	./$(TEST_RUN_PROGRAM)
