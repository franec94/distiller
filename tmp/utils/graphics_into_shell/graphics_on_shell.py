#!/usr/bin/env python3
# -*- enc:utf-8 -*-

"""Python3 Script written for producing plot or graphics directly
within the current shell in which one will run the script, employing
matplotlib submodule plottext.plot in order to achieve the desired goal
tha we are bearing in mind.
"""

from src.libs import *


def main(args):
  """main function which is in charge of orchestrating 
  all the computations needed to produce the final desired and wanted plots.\n
  Params:
  -------
  `args` - plain python dictionary, obtained from parsing argparse.ArgumentParser instance, which keeps input arguments passed in to the script by user.\n
  """
  _ = check_input_file_from_args(args)

  if not os.path.exists(args.output_dir):
    try:
      os.makedirs(args.output_dir)
    except Exception as err:
      if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        print(f"Directory: {args.output_dir} already exists!")
      pass
    pass


  if args.show_graphics:
    # show_both_data_from_filtered_log(args)
    show_stats_data_from_filtered_log(args)
    return
  if args.show_data_from_log:
    if args.input_file:
      show_data_from_filtered_log(args)
  elif args.show_data_from_tb_log:
    if args.input_file:
      show_data_from_tb_log(args)
    pass
  if args.input_file_pruning_trend:
    show_pruning_trend_from_filtered_log(args)
  pass

if __name__ == "__main__":
  parser = get_custom_argparser()
  args = parser.parse_args()
  main(args)
  pass
