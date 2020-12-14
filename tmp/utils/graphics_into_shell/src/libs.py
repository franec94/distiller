# ============================================== #
# Python's Libraries
# ============================================== #
import argparse
import os
import sys

import plotext as plx
try:
    import termplotlib as tpl
    import terminalplot as tp
    import termplot
except:
    pass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
import tensorboard as tb
import numpy as np
import pandas as pd
import tabulate


# ============================================== #
# Custom modules
# ============================================== #

from src.custom_argparser import get_custom_argparser
from src.check_input_args import check_input_file, check_input_file_from_args
from src.show_data_from_filtered_log import show_data_from_filtered_log
from src.show_data_from_tb_log import show_data_from_tb_log
from src.show_pruning_trend import show_pruning_trend_from_filtered_log
from src.show_both_stats import show_both_data_from_filtered_log
from src.show_data_into_shell import show_stats_data_from_filtered_log
