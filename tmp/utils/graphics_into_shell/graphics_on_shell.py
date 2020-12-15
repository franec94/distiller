#!/usr/bin/env python3
# -*- enc:utf-8 -*-

"""Python3 Script written for producing plot or graphics directly
within the current shell in which one will run the script, employing
matplotlib submodule plottext.plot in order to achieve the desired goal
tha we are bearing in mind.
"""

from src.libs import *


def get_main_logger(args):
  """Return custom loggger.
  Args:
  -----
  `args` - python Namespace object with data to correctly setup custom logger.\n
  Return:
  -------
  `logger` - custom logger python object created with support of loggin python module.\n
  """
  
  # create logger
  logger = logging.getLogger('graphics_logger_root')
  logger.setLevel(logging.DEBUG)

  # create file handler which logs even debug messages
  fh = logging.FileHandler('graphics_logger_root.log')
  fh.setLevel(logging.DEBUG)
  
  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)

  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  # add the handlers to the logger
  logger.addHandler(fh)
  logger.addHandler(ch)
  return logger


def main(args):
  """main function which is in charge of orchestrating 
  all the computations needed to produce the final desired and wanted plots.\n
  Params:
  -------
  `args` - plain python dictionary, obtained from parsing argparse.ArgumentParser instance, which keeps input arguments passed in to the script by user.\n
  """
  root_msg_logger = get_main_logger(args)
  root_msg_logger.info(f"Cmnd Line args -> {str(args)}")
  
  root_msg_logger.info()
  _ = check_input_file_from_args(args)
  

  if not os.path.exists(args.output_dir):
    try:
      root_msg_logger.info(f"Creating output dir -> {str(args.output_dir)}")
      os.makedirs(args.output_dir)
    except Exception as err:
      if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        # print(f"Directory: {args.output_dir} already exists!")
        root_msg_logger.info(f"Directory: {args.output_dir} already exists!")
      pass
    pass
  if args.show_graphics:
    # show_both_data_from_filtered_log(args)
    show_stats_data_from_filtered_log(args, msg_logger=root_msg_logger)
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
