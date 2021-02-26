# ========================================== #
# Filename: Makefile_CPD
# Aim: Creating Jpeg datasets.
# ========================================== #

# ======================================= #
# Makefile: Variables
# ======================================= #
INT_SCRIPT = python3

# --------------------------------------- #
# Create Jpeg Dataset Options
# --------------------------------------- #
CJPEG_DS = tmp_create_jpeg_dataset.py 
CJPEG_DS_O_DIR = --output_dir ./out/datasets/script_tool

# Create Jpeg Dataset: Cameramen
# --------------------------------------- #
CJPEG_DS_ARGS_CAMERAME = $(CJPEG_DS_O_DIR)

# Create Jpeg Dataset: test066.png
# --------------------------------------- #
CJPEG_DS_TEST066 = --input_image /home/franec94/Documents/testsets/BSD68/test066.png
CJPEG_DS_ARGS_TEST066 = $(CJPEG_DS_O_DIR) $(CJPEG_DS_TEST066)

# Create Jpeg Dataset: test068.png
# --------------------------------------- #
CJPEG_DS_TEST068 = --input_image /home/franec94/Documents/testsets/BSD68/test068.png
CJPEG_DS_ARGS_TEST068 = $(CJPEG_DS_O_DIR) $(CJPEG_DS_TEST068)

# ======================================= #
# Makefile: Targets
# ======================================= #
c_jpeg_d_HELP:
	clear
	chmod u+x $(CJPEG_DS)
	$(INT_SCRIPT) $(CJPEG_DS) --help

c_jpeg_d_cameramen:
	clear
	chmod u+x $(CJPEG_DS)
	$(INT_SCRIPT) $(CJPEG_DS) $(CJPEG_DS_ARGS_CAMERAME)

c_jpeg_d_test066:
	clear
	chmod u+x $(CJPEG_DS)
	$(INT_SCRIPT) $(CJPEG_DS) $(CJPEG_DS_ARGS_TEST066)

c_jpeg_d_test068:
	clear
	chmod u+x $(CJPEG_DS)
	$(INT_SCRIPT) $(CJPEG_DS) $(CJPEG_DS_ARGS_TEST068)
