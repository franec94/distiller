from src.libraries.std_libs import *


def check_file_exists(a_file_path:str, raise_exception_on_not_exists:bool=False) -> bool:
    """Check whether input file path provided exists.
    Args:
    -----
    `a_file_path` - str object, path to local file system file.\n
    `raise_exception_on_not_exists` - bool object, default to False. If true raises exception when input file passed as input argument to the function does not exists.\n
    Return
    ------
    `bool` - True if exists otherwise False.\n
    """
    if os.path.isfile(a_file_path) is False:
        if raise_exception_on_not_exists:
            raise Exception(f"Error: file '{a_file_path}' not exists!")
        return False
    return True



def check_dir_exists(a_dir_path:str, raise_exception_on_not_exists:bool=False) -> bool:
    """Check whether input file path provided exists.
    Args:
    -----
    `a_dir_path` - str object, path to local file system dir.\n
    `raise_exception_on_not_exists` - bool object, default to False. If true raises exception when input dir passed as input argument to the function does not exists.\n
    Return
    ------
    `bool` - True if exists otherwise False.\n
    """
    if os.path.isdir(a_dir_path) is False:
        if raise_exception_on_not_exists:
            raise Exception(f"Error: dir '{a_dir_path}' not exists!")
        return False
    return True


def read_conf_file_content(conf_file_path: str) -> dict:
    """Read input configuration file content provided to the script as a command line argument.
    Args:
    -----
    `conf_file_path` - str object, path to local file system configuration file, either '.yaml', or '.josn' file kind.
    Return
    ------
    `dict` - containing input configuration options for running the script
    """
    check_file_exists(conf_file_path)

    conf_dict: dict = None
    _, file_ext = os.path.splitext(conf_file_path)

    allowed_conf_file_exts: list = ".yaml,.json".split(",")
    if file_ext.lower() not in allowed_conf_file_exts:
        raise Exception(f"Error: provided conf file's extesion '{file_ext}' is not allowed. Allowed conf file exetsion are: {allowed_conf_file_exts}")

    if file_ext == '.yaml':
        with open(conf_file_path, "r") as conf_fp:
            conf_dict = yaml.load(conf_fp, Loader=yaml.FullLoader)
            # pprint(conf_dict)
            pass
    return conf_dict

def traverse_directory(a_dir_path:str, record_stats) -> collections.namedtuple:
    if not check_dir_exists(a_dir_path): return
    # RecordStats = collections.namedtuple("RecordStats", ["no_dirs", "no_files"])
    keys = ["no_dirs", "no_files"]
    record_stats_local = dict(zip(keys, [0, 0]))
    for dir_name, subdirs_list, files_list in os.walk(a_dir_path):
        record_stats_local["no_dirs"] += 1
        record_stats_local["no_files"] += len(files_list)
        # print('Found directory: %s' % dir_name)
        # print('Found directory: %s' % os.path.basename(dir_name))
        # for file_name in files_list: print('\t%s' % file_name)
        pass
    record_stats["no_dirs"] += record_stats_local["no_dirs"]
    record_stats["no_files"] += record_stats_local["no_files"]
    return record_stats_local


def get_overall_stats_from_input_dirs(conf_dict) -> collections.namedtuple:
    # RecordStats = collections.namedtuple("RecordStats", ["no_dirs", "no_files"])
    keys = ["no_dirs", "no_files"]
    record_stats = dict(zip(keys, [0, 0]))
    for a_dir in conf_dict['input_dirs_list']:
        traverse_directory(a_dir, record_stats)
        pass
    metadata_tbl = dict(
        tabular_data=[list(record_stats.values())],
        headers=list(record_stats.keys()),
    )
    table_stats = tabulate.tabulate(**metadata_tbl)
    print(table_stats)
    return record_stats
