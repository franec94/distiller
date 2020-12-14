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

def compute_regression_curve(x, y, degree=2):
    """Calculate regression curve and retrieve predicted output data from model based on input predictors.
    Params
    ------
    `x` - np.ndarray of input independent variables, e.i. predictors or attributes.\n
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    Return
    ------
    `y_pred` - np.ndarray of predicted data.\n
    """
    print("==> Calculating regression curve...")
    # degree = 2
    # model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # model = LinearRegression().fit(x[:, np.newaxis], y)
    model.fit(x[:, np.newaxis], y)
    y_pred = model.predict(x[:, np.newaxis])
    return y_pred


def show_table(y):
    """Show data collected into a dataframe and then show by means of tabulate python3 module.
    Args
    ----
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    """
    print("==> Show data stats about pruning trend:")
    columns = "Pruning Trend".split(",")
    data_df = pd.DataFrame(y[:np.newaxis], columns=columns)
    # print(data_df.describe())
    res_description = data_df.describe().T
    headers = ['stats'] + list(data_df.T.columns)
    headers = list(data_df.describe().T.columns)
    table_data_dict = dict(
        # tabular_data=res_description, headers=['stats'] + data_df.columns
        tabular_data=res_description, headers=headers, \
        tablefmt="grid"
    )
    table = tabulate.tabulate(**table_data_dict)
    print(table)
    print("==> Last entry recorded:")
    print(tabulate.tabulate(data_df.tail(1)))
    pass

def read_data(args):
    """Read data from source file.
    Return
    ------
    `x` - np.arange for x-axis coordinates.\n
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    """

    filename: str = args.input_file_pruning_trend

    with open(filename, 'r') as f:
        raw_data = list(filter(lambda xx: len(xx) != 0, f.read().split("\n")))
        data = np.array(list(map(float, raw_data)))

    x = np.arange(0, len(data))
    y = data
    return x, y


def plot_graphics(x, y):
    """Plot graphics related to Psrn score.
    Args
    ----
    `x` - np.arange for x-axis coordinates.\n
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    """
    try:
        y_pred = compute_regression_curve(x, y)

        print("==> plot_graphics...")
        plx.scatter(x, y, rows = 17, cols = 70, \
            equations=True, \
            point_color='red', axes=True, \
            point_marker='*', axes_color='')
    
        plx.plot(x, y_pred, rows = 17, cols = 70, \
            equations=True, \
            line_color='blue', axes=True, \
            point_marker='*', axes_color='')
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
    pass


def show_pruning_trend_from_filtered_log(args):
    """Show data from log file.
    Params
    ------
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    """

    x, y = read_data(args)
    show_table(y)
    plot_graphics(x, y)
    pass
