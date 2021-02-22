from src_cpd.libs.std_libs import *


def check_file_exists(a_filepath: str, file_ext: str = None) -> None:
    """Check filepath exists.
    Args:
    -----
    `a_filepath` - str python object, local file system file path to be checked.\n
    `file_ext` - str python object, extension to be checked (Examples: .txt, .csv, ...) Default to None which means not extension will be checked.\n
    """
    if not os.path.exists(a_filepath):
        print(f"Error: resource '{a_filepath}' does not exists!", file=sys.stderr)
        sys.exit(-1)
    if not os.path.isfile(a_filepath):
        print(f"Error: resource '{a_filepath}' is not a file!", file=sys.stderr)
        sys.exit(-2)
    if file_ext:
        _, a_ext = os.path.splitext(a_filepath)
        if a_ext != file_ext:
            print(f"Error: resource '{a_filepath}' is not a '{a_ext}' file!", file=sys.stderr)
            sys.exit(-3)
    pass


def check_dir_exists(a_dirpath: str) -> None:
    """Check filepath exists.
    Args:
    -----
    `a_dirpath` - str python object, local file system file path to be checked.\n
    """
    if not os.path.exists(a_dirpath):
        print(f"Error: resource '{a_dirpath}' does not exists!", file=sys.stderr)
        sys.exit(-1)
    if not os.path.isdir(a_dirpath):
        print(f"Error: resource '{a_dirpath}' is not a directory!", file=sys.stderr)
        sys.exit(-2)
        pass
    pass
