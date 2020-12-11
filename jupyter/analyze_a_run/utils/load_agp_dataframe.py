# ----------------------------------------------- #
# Python's Imports
# ----------------------------------------------- #

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
import yaml

# Plotly imports.
# ----------------------------------------------- #
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.express as px

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

# Data Scienc & Machine Learning main imports.
# ----------------------------------------------- #
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


root_filespaths = "C:\\Users\\Francesco\\Documents\\thesys\\code\\local_projects\\siren-train-logs\\notebooks\\analyze_a_run"
agp_prune_rate_filepath = os.path.join(root_filespaths, "agp-pruning-rate-siren_65_5.txt")
agp_prune_scores_filepath = os.path.join(root_filespaths, "agp-pruning-scores-siren_65_5.txt")

columns_prune_rate = "net.0.linear,net.1.linear,net.2.linear,net.3.linear,net.4.linear,net.5.linear,net.6".split(",")
columns_prune_scores = "mse,psnr,ssim".split(",")

def read_txt_file(text_filepath, columns):
    """Load dataframe with data from input .txt file.
    Args:
    -----
    `text_filepath` - str object, file path to local file system where data are stored within plain .txt file.\n
    `columns` - list object, dataframe target columns.\n
    Return:
    -------
    `data_df` - pd.DataFrame object.\n
    `rows` - python list object with data from which dataframe has been created.\n
    """
    rows = None
    
    # Protocol process txt file
    filter_out_empyt_rows = lambda a_row: len(a_row) != 0
    def map_row_to_list(a_row):
        elms = None
        try:
            a_row_tmp = a_row.strip()
            a_row_tmp = re.sub("( ){1,}", " ", a_row_tmp)
            a_row_tmp = a_row_tmp.replace("\t", " ")
            elms = a_row_tmp.split(" ")
            if len(elms) == 0: return []
            replace_comma_with_dot = lambda a_elm: re.sub(",", ".", a_elm)
            elms = list(map(replace_comma_with_dot, elms))
            elms = list(map(float, elms))
        except:
            return []
        return elms
    filter_out_empyt_lists = lambda a_list: a_list != []

    # Read raw data from .txt file
    with open(text_filepath, "r") as f:
        rows = f.read().split("\n")

    # pprint(rows)
    # Apply Protocol to process txt file
    rows = list(filter(filter_out_empyt_rows, rows))
    # pprint(rows)
    rows = list(map(map_row_to_list, rows))
    # pprint(rows)
    rows = list(filter(filter_out_empyt_lists, rows))
    # pprint(rows)

    data_df = pd.DataFrame(data=rows, columns=columns)
    return data_df, rows

def load_agp_dataframe():
    """Load dataframe with data and scores about AGP-aware pruning.
    Return:
    -------
    `agp_df` - pd.DataFrame object.\n
    """
    agp_df = None

    try:
        data_pr_df, rows_pr = read_txt_file(text_filepath=agp_prune_rate_filepath, columns=columns_prune_rate)
    except Exception as err:
        print(f"An error occurs when processing: {agp_prune_rate_filepath}")
        print(f"{str(err)}")
        pass
    try:
        data_ps_df, rows_ps = read_txt_file(text_filepath=agp_prune_scores_filepath, columns=columns_prune_scores)
    except Exception as err:
        print(f"An error occurs when processing: {agp_prune_scores_filepath}")
        print(f"{str(err)}")
        pass

    joined_columns = list(data_pr_df.columns) + list(data_ps_df.columns)
    agp_df = pd.concat([data_pr_df, data_ps_df], axis=1, names=joined_columns)
    
    return agp_df