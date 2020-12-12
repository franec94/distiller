import argparse
import os
import sys

def get_custom_argparser():
    """Define and retrieve custom parser for collecting input arguments passed in to the script by user.
    Return
    ------
    `parser` - custom instance of argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description="Create text graphics from input data collected earlier within file in textual format or either format such as json, yaml, csv.")
    parser.add_argument('--input_file', dest='input_file', type=str, \
        help='Path to input file which is stored within local file system')
    parser.add_argument('--experiment_id', dest='experiment_id',  type=str, default=None, \
        help='experiment id referring to data stored by means of tensorboard into a proper log file.'
    )
    parser.add_argument('--show_data_from_log', dest='show_data_from_log', action='store_true', default=False, \
        help='set to show data reported within log file and extracted from it.'
    )
    parser.add_argument('--show_data_from_tb_log', dest='show_data_from_tb_log', action='store_true', default=False, \
        help='set to show data reported within tb log file and extracted from it.'
    )
    parser.add_argument('--show_stats_transposed', dest='show_stats_transposed', action='store_true', default=False, \
        help='set to show data reported within log file and extracted from it transposed for displaying reasons.'
    )
    return parser