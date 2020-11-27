# ======================== #
# Variables
# ======================== #

RUN_PROGRAM = run_program.sh
RUN_PROGRAM = run_program_siren_app.sh


# ======================== #
# Targets
# ======================== #

run_program_train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM)
	./$(RUN_PROGRAM)

run_program_siren_app_train_on_cameramen:
	clear
	chmod u+x $(RUN_PROGRAM)
	./$(RUN_PROGRAM)