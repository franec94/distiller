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
from datetime import datetime
from pathlib import Path

# import psycopg2 as ps
import argparse
import collections
import contextlib
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
import subprocess
import time
import tabulate
import yaml
import zipfile

try:
    from googleapiclient.http import MediaIoBaseDownload
except Exception as err:
    print(f"{str(err)}")
    pass
