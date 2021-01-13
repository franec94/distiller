from __future__ import print_function

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
import tabulate
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--output_file_path", dest="output_file_path", type=str, required=True,\
    help="Path whitin local file system where output table will be saved."
)
parser.add_argument("--a_row", dest="a_row", type=str, required=True,\
    help="Row to be saved as record of output table."
)
parser.add_argument("--headers", dest="headers", type=str, required=True,\
    help="Headers to be used for output table."
)
parser.add_argument("--row_sep", dest="row_sep", type=str, required=False, default="," \
    help="Row separator special character upon whihch executing split to retrieve row fields. (Default: ',')"
)
parser.add_argument("--header_sep", dest="row_sep", type=str, required=False, default="|" \
    help="Headers separator special character upon whihch executing split to retrieve row fields. (Default: '|')"
)

def main(args):
    a_row = args.a_row.split(f"{args.row_sep}")
    headers = args.headers.split(f"{args.header_sep}")

    if len(a_row) == 0 or len(headers) == 0:
        print("Input row is empty, no processing is done.")
        sys.exit(0)
    if len(headers) == 0:
        print("Input headers list is empty, no processing is done.")
        sys.exit(0)
    if len(headers) != len(a_row):
        if len(a_row) == 0 or len(headers) == 0:
        print(f"Error: len(a_row) != len(headers), that is {len(a_row)} != {len(headers)}", file=sys.stderr)
        sys.exit(-1)

    metadat_table_dict = dict(
        tabular_data=[a_row],
        headers=headers,
        tablefmt="github"
    )
    
    out_table = tabulate.tabulate(**metadat_table_dict)
    with open(args.output_file_path, "w") as out_fp_table:
        out_fp_table.write(out_table)
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass