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
    # print("==> Calculating regression curve...")
    # degree = 2
    # model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # model = LinearRegression().fit(x[:, np.newaxis], y)
    model.fit(x[:, np.newaxis], y)
    y_pred = model.predict(x[:, np.newaxis])
    return y_pred


def read_data(args):
    """Read data from source file.
    Return
    ------
    `x` - np.arange for x-axis coordinates.\n
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    """

    filename_1: str = args.input_file
    filename_2: str = args.input_file_pruning_trend

    with open(filename_1, 'r') as f:
        raw_data = list(filter(lambda xx: len(xx) != 0, f.read().split("\n")))
        data = np.array(list(map(float, raw_data)))

    x = np.arange(0, len(data))
    y = data
    
    with open(filename_2, 'r') as f:
        raw_data = list(filter(lambda xx: len(xx) != 0, f.read().split("\n")))
        data = np.array(list(map(float, raw_data)))

    x_2 = np.arange(0, len(data))
    y_2 = data
    return x, y, x_2, y_2


def show_table(y, y_2, labels):
    """Show data collected into a dataframe and then show by means of tabulate python3 module.
    Args
    ----
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    """
    # print("==> Show data stats:")
    columns = f"{labels},bpp".split(",")
    n_hf, n_hl = 64, 5
    w, h = 256, 256
    baseline_size = (n_hf*2 + 2) + (n_hf **2 * n_hl + n_hf * n_hl) + (n_hf + 1)

    y_3 = (baseline_size - y_2 * baseline_size/100) * 32 / (w * h)
    y_tmp = np.array(list(zip(y, y_2, y_3)))
    # data_df = pd.DataFrame(y_tmp[:np.newaxis], columns=columns)
    data_df = pd.DataFrame(y_tmp, columns=columns)
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
    print(tabulate.tabulate(data_df.tail(3),headers=data_df.columns))
    return data_df


def plot_graphics_into_shell_via_plotext(args, data_df):
    x = np.arange(0, data_df.shape[0])
    if args.show_bpp_trend:
        plx.plot(x, data_df['bpp'].values, rows = 17, cols = 70, \
            equations=True, \
            point_color='red', axes=True, \
            point_marker='+', axes_color='',)
        plx.plot(x, y_pred_3, rows = 17, cols = 70, \
            equations=True, \
            line_color='yellow', axes=True, \
            point_marker='+', axes_color='',)
        plx.show()
    elif args.show_psnr_trend:
        plx.scatter(x, y, rows = 17, cols = 70, \
            equations=True, \
            point_color='red', axes=True, \
            point_marker='*', axes_color='')
    
        plx.plot(x, y_pred, rows = 17, cols = 70, \
            equations=True, \
            line_color='blue', axes=True, \
            point_marker='*', axes_color='')
        plx.show()
        pass
    elif args.show_psnr_vs_bpp:
        # plx.scatter(data_df['bpp'].values, data_df['Psnr score'].values, rows = 17, cols = 70, \
        plx.scatter(data_df['bpp'].values, y, rows = 17, cols = 70, \
            equations=True, \
            point_color='red', axes=True, \
            point_marker='*', axes_color='')

        y_psnr_pred = compute_regression_curve(
            np.array(list(data_df['bpp'].values)),
            # np.array(list(data_df['Psnr score'].values)))
            y
            )
    
        plx.plot(data_df['bpp'].values, y_psnr_pred, rows = 17, cols = 70, \
            equations=True, \
            line_color='blue', axes=True, \
            point_marker='+', axes_color='',)
        plx.show()
        pass
    elif args.show_prune_trend:
        plx.scatter(x_2, y_2, rows = 17, cols = 70, \
            equations=True, \
            point_color='orange', axes=True, \
            point_marker='*', axes_color='')
    
        plx.plot(x, y_pred_2, rows = 17, cols = 70, \
            equations=True, \
            line_color='green', axes=True, \
            point_marker='+', axes_color='',)
        plx.show()
    else:
        # print("==> plot_graphics...")
        plx.scatter(x, y, rows = 17, cols = 70, \
            equations=True, \
            point_color='red', axes=True, \
            point_marker='*', axes_color='')
    
        plx.plot(x, y_pred, rows = 17, cols = 70, \
            equations=True, \
            line_color='blue', axes=True, \
            point_marker='*', axes_color='')
        
        plx.scatter(x_2, y_2, rows = 17, cols = 70, \
            equations=True, \
            point_color='orange', axes=True, \
            point_marker='*', axes_color='')
    
        plx.plot(x, y_pred_2, rows = 17, cols = 70, \
            equations=True, \
            line_color='green', axes=True, \
            point_marker='+', axes_color='',)
        plx.plot(x, y_pred_3, rows = 17, cols = 70, \
            equations=True, \
            line_color='yellow', axes=True, \
            point_marker='+', axes_color='',)
        plx.show()
        pass
    pass


def plot_graphics_v2(args, data_df):
    data_df
    y_pred = compute_regression_curve(x, y)
    y_pred_2 = compute_regression_curve(x_2, y_2)
    y_pred_3 = compute_regression_curve(x, data_df['bpp'].values)

    plot_graphics_into_shell_via_plotext(args, data_df)

    pass

def plot_graphics(args, x, y, x_2, y_2, data_df):
    """Plot graphics related to Psrn score.
    Args
    ----
    `x` - np.arange for x-axis coordinates.\n
    `y` - np.ndarray for y-axis coordinates, Psnr Scores.\n
    `x_2` - np.arange for x-axis coordinates.\n
    `y_2` - np.ndarray for y-axis coordinates, Prune trend.\n
    """
    try:
        y_pred = compute_regression_curve(x, y)
        y_pred_2 = compute_regression_curve(x_2, y_2)
        y_pred_3 = compute_regression_curve(x, data_df['bpp'].values)

    
        if args.show_bpp_trend:
            plx.plot(x, data_df['bpp'].values, rows = 17, cols = 70, \
                equations=True, \
                point_color='red', axes=True, \
                point_marker='+', axes_color='',)
            plx.plot(x, y_pred_3, rows = 17, cols = 70, \
                equations=True, \
                line_color='yellow', axes=True, \
                point_marker='+', axes_color='',)
            plx.show()
        elif args.show_psnr_trend:
            plx.scatter(x, y, rows = 17, cols = 70, \
                equations=True, \
                point_color='red', axes=True, \
                point_marker='*', axes_color='')
        
            plx.plot(x, y_pred, rows = 17, cols = 70, \
                equations=True, \
                line_color='blue', axes=True, \
                point_marker='*', axes_color='')
            plx.show()
            pass
        elif args.show_psnr_vs_bpp:
            # plx.scatter(data_df['bpp'].values, data_df['Psnr score'].values, rows = 17, cols = 70, \
            plx.scatter(data_df['bpp'].values, y, rows = 17, cols = 70, \
                equations=True, \
                point_color='red', axes=True, \
                point_marker='*', axes_color='')

            y_psnr_pred = compute_regression_curve(
                np.array(list(data_df['bpp'].values)),
                # np.array(list(data_df['Psnr score'].values)))
                y
                )
        
            plx.plot(data_df['bpp'].values, y_psnr_pred, rows = 17, cols = 70, \
                equations=True, \
                line_color='blue', axes=True, \
                point_marker='+', axes_color='',)
            plx.show()
            pass
        elif args.show_prune_trend:
            plx.scatter(x_2, y_2, rows = 17, cols = 70, \
                equations=True, \
                point_color='orange', axes=True, \
                point_marker='*', axes_color='')

            plx.plot(x, y_pred_2, rows = 17, cols = 70, \
                equations=True, \
                line_color='green', axes=True, \
                point_marker='+', axes_color='',)
            plx.show()
        else:
            # print("==> plot_graphics...")
            plx.scatter(x, y, rows = 17, cols = 70, \
                equations=True, \
                point_color='red', axes=True, \
                point_marker='*', axes_color='')

            plx.plot(x, y_pred, rows = 17, cols = 70, \
                equations=True, \
                line_color='blue', axes=True, \
                point_marker='*', axes_color='')
            
            plx.scatter(x_2, y_2, rows = 17, cols = 70, \
                equations=True, \
                point_color='orange', axes=True, \
                point_marker='*', axes_color='')

            plx.plot(x, y_pred_2, rows = 17, cols = 70, \
                equations=True, \
                line_color='green', axes=True, \
                point_marker='+', axes_color='',)
            plx.plot(x, y_pred_3, rows = 17, cols = 70, \
                equations=True, \
                line_color='yellow', axes=True, \
                point_marker='+', axes_color='',)
            plx.show()
            pass
        pass
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


def show_both_data_from_filtered_log(args):
    """Show data from log file.
    Params
    ------
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    """

    x, y, x_2, y_2 = read_data(args)
    # show_table(y, label='Psnr score')
    # show_table(y_2, label='Prune trend')
    
    data_df = show_table(y, y_2, labels='Psnr score,Prune trend')
    data_df.to_csv("data.csv")

    # plot_graphics(args, x, y, x_2, y_2, data_df)
    plot_graphics_v2(args, data_df)

    pass