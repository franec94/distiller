from src_cpd.libs.std_libs import *


def read_conf_file(conf_file_path, raise_exception = True):
    """Read config `.yaml` file."""
    _ = check_file_exists(a_filepath=conf_file_path, file_ext=None, raise_exception=raise_exception)
    try:
        with open(conf_file_path, "r") as f:
            try:
                conf_data = yaml.load(f, Loader=yaml.FullLoader)
            except:
                conf_data = yaml.load(f)
                pass
            pass
        return conf_data
    except Exception as err:
        raise Exception(f"Error: when reading input conf file '{conf_file_path}'\n{str(err)}")
    pass


def check_file_exists(a_filepath: str, file_ext: str = None, raise_exception=False) -> None:
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
