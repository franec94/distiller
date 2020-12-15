# ============================================== #
# Python's Libraries
# ============================================== #
from pprint import pprint

import argparse
import os
import sys
import collections

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

# -------------------------------- #
# Prepare Data Functions
# -------------------------------- #
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
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x[:, np.newaxis], y)
    y_pred = model.predict(x[:, np.newaxis])
    return y_pred


def read_data(filename: str) -> np.array:
    """Read data from source file.
    Args
    ----
    `filename` - str object filename path.\n
    Return
    ------
    `data_array` - np.array.\n
    """

    data_array: np.array = None
    with open(filename, 'r') as f:
        raw_data = list(filter(lambda xx: len(xx) != 0, f.read().split("\n")))
        data_array = np.array(list(map(float, raw_data)))
    return data_array


def create_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Create reference dataframe for Psnr,Prune rate and BPP score.
    Args
    ----
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    Return
    ------
    `data_df` - pd.DataFrame with data.\n
    """
    data_df: pd.DataFrame = None

    data_array_psnr = read_data(filename=args.input_file)
    # pprint(data_array_psnr[:5])
    data_array_prune_rate = read_data(filename=args.input_file_pruning_trend)
    # pprint(data_array_prune_rate[:5])

    n_hf, n_hl = 64, 5
    w, h = 256, 256
    baseline_size = (n_hf * 2 + 2) + (n_hf **2 * n_hl + n_hf * n_hl) + (n_hf + 1)
    data_array_bpp = (baseline_size - data_array_prune_rate * baseline_size/100) * 32 / (w * h)

    map_to_ord_dict = lambda item: collections.OrderedDict(
        psnr_score=item[0],
        prune_rate=item[1],
        bpp=item[2]
    )
    data_to_zip = [data_array_psnr, data_array_prune_rate, data_array_bpp]
    data_for_df = list(
        map(map_to_ord_dict, zip(*data_to_zip))
    )
    columns="Psnr Score,Prune Rate,bpp".split(",")
    data_df = pd.DataFrame(data = data_for_df)
    data_df.columns = columns
    return data_df


def create_dataframe_reg_curves(data_df: pd.DataFrame) -> pd.DataFrame:
    """Create reference dataframe for Psnr,Prune rate and BPP score.
    Args
    ----
    `data_df` - pd.DataFrame with data for psnr,prune rate, and bpp scores.\n
    Return
    ------
    `data_reg_df` - pd.DataFrame with data about reg curves for psnr, prune rate and bpp.\n
    """
    data_reg_df: pd.DataFrame = None

    x = np.arange(0, data_df.shape[0])

    y_pred_psnr = compute_regression_curve(x, data_df['Psnr Score'].values)
    y_pred_prune_rate = compute_regression_curve(x, data_df['Prune Rate'].values)
    y_pred_bpp = compute_regression_curve(x, data_df['bpp'].values)

    map_to_ord_dict = lambda item: collections.OrderedDict(
        y_pred_psnr=item[0],
        y_pred_prune_rate=item[1],
        y_pred_bpp=item[2]
    )
    data_for_df = list(map(map_to_ord_dict, zip(y_pred_psnr, y_pred_prune_rate, y_pred_bpp)))
    columns="y_pred_psnr,y_pred_prune_rate,y_pred_bpp".split(",")
    data_reg_df = pd.DataFrame(data = data_for_df, columns=columns)

    return data_reg_df

# -------------------------------- #
# Show Data Functions
# -------------------------------- #
def show_table_stats(data_df: pd.DataFrame) -> None:
    """Show data collected into a dataframe and then show by means of tabulate python3 module.
    Args
    ----
    `data_df` - pd.DataFrame.\n
    """
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
    print("==> Last entry(ies) recorded:")
    print(tabulate.tabulate(data_df.tail(3),headers=data_df.columns))
    pass


def show_data(x, y, y_pred, **kwargs):
    """Show data into shell.
    Args
    ----
    `x` - np.array or list, containing data for x-axis.\n
    `y` - np.array or list, containing data for y-axis to plot scatter data.\n
    `y_pred` - np.array or list, containing data for y-axis to plot reg curve.\n
    `**kwargs` -  data pairs key-value for plx.plot.\n
    """

    # point_color= kwargs['point_color'] if hasattr(kwargs, 'point_color') else 'red'
    # line_color= kwargs['line_color'] if hasattr(kwargs, 'line_color') else 'blue'

    point_color= kwargs['point_color']
    line_color= kwargs['line_color']
    point_marker = kwargs['point_marker']

    # Scatter plot
    plx.scatter(x, y, rows = 17, cols = 70, \
        equations=True, \
        point_color=point_color, axes=True, \
        point_marker=point_marker, axes_color='',)
    
    # line plot plot
    plx.plot(x, y_pred, rows = 17, cols = 70, \
        equations=True, \
        line_color=line_color, axes=True, \
        axes_color='',)
    pass


def show_psnr_data(data_df: pd.DataFrame) -> None:
    """Show data psnr data into shell.
    Args
    ----
    `data_df` - pd.DataFrame or list, containing data for x-axis.\n
    """
    y = data_df['Psnr Score'].values
    y_pred = data_df['y_pred_psnr'].values
    data_to_show: collections.OrderedDict = collections.OrderedDict(
        x=np.arange(len(y)),
        y=y,
        y_pred=y_pred,
        point_color='red',
        line_color='blue',
        point_marker='*',
    )
    show_data(**data_to_show)
    pass


def show_bpp_data(data_df):
    """Show data bpp data into shell.
    Args
    ----
    `data_df` - pd.DataFrame or list, containing data for x-axis.\n
    """
    y = data_df['bpp'].values
    y_pred = data_df['y_pred_bpp'].values
    data_to_show: collections.OrderedDict = collections.OrderedDict(
        x=np.arange(len(y)),
        y=y,
        y_pred=y_pred,
        point_color='red',
        line_color='yellow',
        point_marker='+',

    )
    show_data(**data_to_show)
    pass


def show_prune_rate_data(data_df):
    """Show data prune rate data into shell.
    Args
    ----
    `data_df` - pd.DataFrame or list, containing data for x-axis.\n
    """
    y = data_df['Prune Rate'].values
    y_pred = data_df['y_pred_prune_rate'].values
    data_to_show: collections.OrderedDict = collections.OrderedDict(
        x=np.arange(len(y)),
        y=y,
        y_pred=y_pred,
        point_color='orange',
        line_color='green',
        point_marker='o',
    )
    show_data(**data_to_show)
    pass


def plot_graphics(args, data_df):
    """Plot graphics related to Psrn score, Prune Rate and Bpp.
    Args
    ----
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    `data_df` - pd.DataFrame with data.\n
    """
    try:
        if args.show_bpp_trend:
            show_bpp_data(data_df)
            plx.show()
        elif args.show_psnr_trend:
            show_psnr_data(data_df)
            plx.show()
            pass
        elif args.show_psnr_vs_bpp:
            x, y = np.array(list(data_df['bpp'].values)), np.array(list(data_df['Psnr Score'].values))
            y_psnr_pred = compute_regression_curve(x, y)

            # plx.scatter(data_df['bpp'].values, data_df['Psnr score'].values, rows = 17, cols = 70, \
            plx.scatter(x, y, rows = 17, cols = 70, \
                equations=True, \
                point_color='red', axes=True, \
                point_marker='*', axes_color='')
            plx.plot(x, y_psnr_pred, rows = 17, cols = 70, \
                equations=True, \
                line_color='blue', axes=True, \
                point_marker='+', axes_color='',)
            plx.show()
            pass
        elif args.show_prune_trend:
            show_prune_rate_data(data_df)
            plx.show()
        else:
            show_psnr_data(data_df)
            show_prune_rate_data(data_df)
            show_bpp_data(data_df)
            plx.show()
        pass

    except Exception as err:
        print(f"Error: occurred when plotting data via plotext.\n{str(err)}")
        pass

    pass


def show_stats_data_from_filtered_log(args: argparse.Namespace, msg_logger=None) -> None:
    """Show data from log file.
    Params
    ------
    `args` - Namespace object keeping input arguments passed in to the script by user.\n
    `msg_logger` - custom logger.\n
    """

    # Create base dataframe
    # and save it.
    msg_logger.info(f"Creating base line dataframe, reading data from: {args.input_file} and {args.input_file_pruning_trend}...")
    data_df: pd.DataFrame = create_dataframe(args=args)
    msg_logger.info(f"Task Done.")

    
    msg_logger.info(f"Saving base line dataframe.")
    data_dest_filepath = os.path.join(args.output_dir, 'data.csv')
    data_df.to_csv(f"{data_dest_filepath}")
    msg_logger.info(f"Task Done.")
    
    msg_logger.info(f"Showing base line dataframe contente as overall statistics.")
    show_table_stats(data_df=data_df)
    msg_logger.info(f"Task Done.")

    # Create regression curves dataframe
    msg_logger.info(f"Creating dataframe with regression curve data...")
    data_reg_df: pd.DataFrame = create_dataframe_reg_curves(data_df=data_df)    
    msg_logger.info(f"Task Done.")

    msg_logger.info(f"Join by columns resulting dataframes, earlier created...")
    joined_columns = list(data_df.columns) + list(data_reg_df.columns)
    data_df = pd.concat([data_df, data_reg_df], axis=1, names=joined_columns)
    msg_logger.info(f"Task Done.")


    msg_logger.info(f"Plot Data...")
    plot_graphics(args, data_df)
    msg_logger.info(f"Task Done.")

    pass
