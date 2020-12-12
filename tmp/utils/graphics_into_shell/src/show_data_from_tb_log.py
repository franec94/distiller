# ============================================== #
# Python's Libraries
# ============================================== #
import argparse
import os
import sys

import plotext as plx
import termplotlib as tpl
import terminalplot as tp
import termplot

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
import tensorboard as tb
import numpy as np
import pandas as pd
import tabulate


def compute_regression_curve(x):
    """Calculate regression curve and retrieve predicted output data from model based on input predictors.
    Params
    ------
    `x` - np.ndarray of input independent variables, e.i. predictors or attributes.\n
    Return
    ------
    `y_pred` - np.ndarray of predicted data.\n
    """
    print("==> Calculating regression curve...")
    degree = 2
    # model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # model = LinearRegression().fit(x[:, np.newaxis], y)
    model.fit(x[:, np.newaxis], y)
    y_pred = model.predict(x[:, np.newaxis])
    return y_pred

def show_data_from_tb_log(args):
    """Show data from tb log file.
    Params
    ------
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    """
    
    if args.experiment_id is None: return

    experiment_id = args.experiment_id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    data_df = experiment.get_scalars()

    if args.show_stats_transposed:
        res_description = data_df.describe().T
        headers = ['stats'] + list(data_df.T.columns)
        headers = list(data_df.describe().T.columns)
    else:
        res_description = data_df.describe()
        headers = ['stats'] + list(data_df.columns)
        headers = list(data_df.describe().columns)
    
    table_data_dict = dict(
        # tabular_data=res_description, headers=['stats'] + data_df.columns
        tabular_data=res_description, headers=headers, \
        tablefmt="grid"
    )
    table = tabulate.tabulate(**table_data_dict)
    print(table)
    print("==> Last entry recorded:")
    print(tabulate.tabulate(data_df.tail(1)))

    """
    try:
        plx.scatter(x, y, rows = 17, cols = 70, \
        equations=True, \
        point_color='red', axes=True, \
        point_marker='*', axes_color='')
        print("==> Calculating regression line...")

        degree = 2
        # model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        # model = LinearRegression().fit(x[:, np.newaxis], y)
        model.fit(x[:, np.newaxis], y)
        y_pred = model.predict(x[:, np.newaxis])
        plx.plot(x, y_pred, rows = 17, cols = 70, \
        equations=True, \
        line_color='blue', axes=True, \
        point_marker='*', axes_color='')
        # plx.set_xlim([0, len(x)])
        # plx.set_ylim([min(x), max(x)])
        plx.show()
    except Exception as err:
        print(f"Error: occurred when plotting data via plotext.\n{str(err)}")
        pass

    try:
        fig = tpl.figure()
        fig.plot(x, y, width=60, height=20)
        fig.show()
    except Exception as err:
        # print(f"Error: occurred when plotting data via termplotlib.\n{str(err)}")
        pass

    try:
        termplot.plot(x, y)
    except Exception as err:
        # print(f"Error: occurred when plotting data via termplot.\n{str(err)}")
        pass

    try:
        # tp.plot(list(x),list(y))
        pass
    except Exception as err:
        print(f"Error: occurred when plotting data via terminalplot.\n{str(err)}")
        pass
    """
    pass
