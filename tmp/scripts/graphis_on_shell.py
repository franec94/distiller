#!/usr/bin/env python3
# -*- enc:utf-8 -*-

"""Python3 Script written for producing plot or graphics directly
within the current shell in which one will run the script, employing
matplotlib submodule plottext.plot in order to achieve the desired goal
tha we are bearing in mind.
"""

import argparse
import os
import sys

import plotext as plx
import termplotlib as tpl
import terminalplot as tp
import termplot
import numpy as np

parser = argparse.ArgumentParser(description="Create text graphics from input data collected earlier within file in textual format or either format such as json, yaml, csv.")
parser.add_argument('--input_file', dest='input_file', type=str, \
  help='Path to input file which is stored within local file system')


def check_input_file(filename: str, return_bool=False) -> bool:
  if not os.path.exists(filename):
    pass
  if not os.path.isfile(filename):
    print(f"Error: input resources '{filename}' passed in as file, is not a file!")
    if return_bool: False
    sys.exit(-1)
    pass

  file_basename = os.path.basename(filename)
  _, ext = os.path.splitext(file_basename)
  if ext not in ".txt,.csv,.json,.yaml".split(","):
    allowed_ext: list = ".txt,.csv,.json,.yaml".split(",")
    print(f"Error: input resources '{filename}' passed in as f{ext} file, which is not allowed, while are allowed: {str(allowed_ext)}")
    if return_bool: False
    sys.exit(-1)
    pass

  return True


def check_input_file_from_args(args) -> None:
  filename: str = args.input_file
  is_right_file = check_input_file(filename=filename)
  return is_right_file



def main(args):
  """main function which is in charge of orchestrating 
  all the computations needed to produce the final desired and wanted plots.\n
  Params:
  -------
  `args` - plain python dictionary, obtained from parsing argparse.ArgumentParser instance, which keeps input arguments passed in to the script by user.\n
  """
  _ = check_input_file_from_args(args)
  filename: str = args.input_file

  with open(filename, 'r') as f:
    raw_data = list(filter(lambda xx: len(xx) != 0, f.read().split("\n")))
    data = np.array(list(map(float, raw_data)))

  x = np.arange(0, len(data))
  y = data
  try:
    plx.scatter(x, y, rows = 17, cols = 70, \
      equations=True, \
      point_color='red', axes=True, \
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

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
  pass
