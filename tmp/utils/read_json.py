#!/usr/bin/env python3
# -*- enc:utf-8 -*-
from pprint import pprint

import json
import argparse
import os
import sys
import tabulate

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input_filename", dest="input_filename", type=str,
    help="json input filename"
)
parser.add_argument("--show_as_table", dest="show_as_table", action="store_true", default=False,
    help="Show data as table"
)
parser.add_argument("--save_as_csv", dest="save_as_csv", action="store_true", default=False,
    help="Save data into csv file, within local file system"
)


def from_dict_2_df(data_dict: dict) -> pd.DataFrame:
    """Convert data from plain python dictionary to pandas dataframe object."""
    def add_layer_name_as_info(item):
        k, v = item
        v['layer'] = k
        return v
    layer_dicts_list = list(map(add_layer_name_as_info, data_dict.items()))
    df  = pd.DataFrame(layer_dicts_list)
    return df
    
def show_as_table_via_tabulate(data_df: pd.DataFrame, save_to_file: bool =False) -> None:
    """Show data as table via tabulate python's package.
        https://pypi.org/project/tabulate/
    """
    tabulate_info_dict = dict(
        tabular_data=data_df,
        headers=data_df.columns,
        tablefmt="grid"
    )
    table = tabulate.tabulate(**tabulate_info_dict)
    print(table)
    if save_to_file:
        try:
            with open("grid_table.txt", "w") as f:
                f.write(table)
                pass
        except Exception as err:
            print(f"Error: {str(err)}. Error occurred when attempting to save to local file system table content.")
            pass
    pass

def show_data(data_dict: dict, args) -> None:
    """Show data to standard output as either formatted table or plain dictionary object via pprint."""
    data_df: pd.DataFrame = from_dict_2_df(data_dict=data_dict)
    if args.show_as_table:
        show_as_table_via_tabulate(data_df=data_df, save_to_file=True)
        pass
    else:
        pprint(data_df)
        pass
    pass


def check_input_filename(args):
    """Check whether user provided input file satisfyes requested and constraints to be a json resource file."""
    filename = args.input_filename
    file_basename = os.path.basename(filename)
    
    _, ext = os.path.splitext(file_basename)
    if ext != ".json":
        print(f"Error: input file should be '.json' file, instead found '{ext}'")
        sys.exit(-1)
        pass
    if not os.path.isfile(filename):
        print(f"Error: input resource '{filename}' is not a file!")
        sys.exit(-1)
        pass
    pass


def get_data_from_json_filename(filename) -> dict:
    """Read input data from json file and store content into plain python dictionary instance."""
    data: dict = {}
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def main(args):
    """Process input file."""
    
    check_input_filename(args)
    filename = args.input_filename
    
    data_dict: dict = \
        get_data_from_json_filename(filename)
        
    show_data(data_dict=data_dict, args=args)
    
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass