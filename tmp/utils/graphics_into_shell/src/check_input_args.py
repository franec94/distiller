import os
import sys

def check_input_file(filename: str, return_bool=False) -> bool:
  """Check whether input filename is a valid file."""
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
  """Check input filename provided by Namespace instance, retrieved from parsed input arguments."""
  filename: str = args.input_file
  if filename:
    is_right_file = check_input_file(filename=filename)
    return is_right_file
  return False

