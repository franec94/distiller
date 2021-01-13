from __future__ import print_function

# ========================================================================== #
# Globals
# ========================================================================== # 
SHOW_VISDOM_RESULTS = False
DARK_BACKGROUND_PLT = True
SHOW_RESULTS_BY_TABS = True
DASH_TEMPLATES_LIST = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]

# ========================================================================== #
# Python's Imports
# ========================================================================== #

# Std Lib and others.
# ----------------------------------------------- #
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
# from contextlib import closing
from io import BytesIO
from PIL import Image
from pprint import pprint

# import psycopg2 as ps
import argparse
import contextlib
import collections
import copy
import datetime
import functools
import glob
import itertools
import json
import operator
import os
import pathlib
import pickle
import re
import sqlite3
import shutil
import sys
import time
import tabulate
import yaml

# Plotly imports.
# ----------------------------------------------- #
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.express as px

# Dash imports.
# ----------------------------------------------- #
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# FastAi imports.
# ----------------------------------------------- #
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.utils import *
from fastcore.test import *
from nbdev.showdoc import *
from fastcore.dispatch import typedispatch
from functools import partial
import inspect

from fastcore.imports import in_notebook, in_colab, in_ipython

# Constraint imports.
# ----------------------------------------------- #
if in_colab():
    from google.colab import files
    pass

if  (in_notebook() or in_ipython()) and SHOW_VISDOM_RESULTS:
    import visdom
    pass

if in_colab() or in_notebook() or in_colab():
    # Back end of ipywidgets.
    from IPython.display import display    
    import ipywidgets as widgets
    pass

# Data Scienc & Machine Learning main imports.
# ----------------------------------------------- #
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if DARK_BACKGROUND_PLT:
    # plt.style.use('dark_background')
    # plt.style.use('ggplot')
    # plt.style.use('seaborn')
    pass
import seaborn as sns # sns.set_theme(style="white") # sns.set(style="whitegrid", color_codes=True)
# sns.set(style="darkgrid", color_codes=True)

# skimage
# ----------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# sklearn
# ----------------------------------------------- #
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import ParameterGrid

# ========================================================================== #
# Custom Imports
# ========================================================================== #

# ----------------------------------------------- #
# Custom Import: generics 
# ----------------------------------------------- #

# functiions - imports # from utils.generics.functions
from utils.generics.functions import check_dir_exists
from utils.generics.functions import get_all_files_by_ext
from utils.generics.functions import laod_data_from_files_list
from utils.generics.functions import check_file_exists
from utils.generics.functions import read_conf_file
from utils.generics.functions import load_target_image
from utils.generics.functions import get_dict_dataframes
from utils.generics.functions import get_dataframe

# ----------------------------------------------- #
# Custom Import: graphics
# ----------------------------------------------- # 
from utils.graphics.make_graphics import compare_compressions
from utils.graphics.custom_show_data import show_summary_data_agp

from utils.handle_dataframes import prepare_and_merge_target_dfs, calculate_several_jpeg_compression, get_cropped_by_center_image

# ----------------------------------------------- #
# handle_server_connections 
# ----------------------------------------------- #
from utils.handle_server_connections.work import fetch_data
from utils.handle_server_connections.work import fetch_data_by_status
from utils.handle_server_connections.work import fetch_data_by_constraints
from utils.handle_server_connections.work import get_info_from_logged_parser
from utils.handle_server_connections.work import insert_data_read_from_logs

from utils.handle_server_connections.handle_server_connection import get_data_from_db
from utils.handle_server_connections.handle_server_connection import get_data_from_db_by_status
from utils.handle_server_connections.handle_server_connection import get_constraints_for_query_db

from utils.handle_server_connections.db_tables import TableRunsDetailsClass
from utils.handle_server_connections.db_tables import TableRunsDetailsTupleClass

# ----------------------------------------------- #
# Custom Import: custom_dash_app
# ----------------------------------------------- # 
from utils.custom_dash_app.custom_dash_app import get_dash_app

# ----------------------------------------------- #
# Custom Import: data_loaders
# ----------------------------------------------- # 
from utils.loaders.load_agp_dataframe import load_agp_dataframe
from utils.loaders.load_quant_aware_trained_data import load_siren_quant_aware_data_df
from utils.data_loaders.data_loader import load_data_baseline, load_jpeg_baseline
