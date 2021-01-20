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

    # conf_dict: dict = None
    _, file_ext = os.path.splitext(conf_file_path)

    allowed_conf_file_exts: list = ".yaml,.json".split(",")
    if file_ext.lower() not in allowed_conf_file_exts:
        raise Exception(f"Error: provided conf file's extesion '{file_ext}' is not allowed. Allowed conf file exetsion are: {allowed_conf_file_exts}")

    if file_ext == '.yaml':
        with open(conf_file_path, "r") as conf_fp:
            # conf_dict = yaml.load(conf_fp, Loader=yaml.FullLoader)
            conf_dict = yaml.load(conf_fp)
            # pprint(conf_dict)
            # print(type(conf_dict))
            data = collections.OrderedDict(
                error=f"conf_dict object is not 'dict', but {type(conf_dict)}",
                src_file=conf_file_path,
            )
            meta_tb = dict(
                tabular_data=data.items()
            )
            table = tabulate.tabulate(**meta_tb)
            assert type(conf_dict) == dict, f"\n{str(table)}\n{str(conf_dict)}"
            # sys.exit(0)
            pass
    return conf_dict


def traverse_directory(a_dir_path:str, record_stats, keys) -> collections.namedtuple:
    if not check_dir_exists(a_dir_path): return
    # RecordStats = collections.namedtuple("RecordStats", ["no_dirs", "no_files"])
    
    record_stats_local = dict(zip(keys, [0] * len(keys)))

    for dir_name, subdirs_list, files_list in os.walk(a_dir_path):
        record_stats_local["no_dirs"] += 1
        record_stats_local["no_files"] += len(files_list)

        if dir_name.endswith("configs"): record_stats_local["no_configs_dirs"] += 1
        else: record_stats_local["no_trains_dirs"] += 1


        models_files_list = list(filter(lambda item: item.endswith(".tar"), files_list))
        record_stats_local["no_models_files"] += len(models_files_list)

        def get_final_model(item):
            file_name = os.path.basename(item)
            return file_name.startswith("_final")
        final_model = len(list(filter(get_final_model, models_files_list)))
        record_stats_local["no_finals_models"] += final_model

        def get_ckpt_model(item):
            file_name = os.path.basename(item)
            return file_name.startswith("_checkpoint")
        ckpt_model = len(list(filter(get_ckpt_model, models_files_list)))
        def get_intermediate_model_model(item):
            file_name = os.path.basename(item)
            return file_name.startswith("_pruned")
        
        intermediate_models = len(list(filter(get_intermediate_model_model, models_files_list)))
        record_stats_local["no_pruned_models"] += intermediate_models

        models_files_list = final_model + intermediate_models + ckpt_model
        record_stats_local["no_models_to_be_tested"] += models_files_list

        # print('Found directory: %s' % dir_name)
        # print('Found directory: %s' % os.path.basename(dir_name))
        # for file_name in files_list: print('\t%s' % file_name)
        pass
    for k, v in record_stats_local.items():
        record_stats[f"{k}"] += v
        # record_stats["no_dirs"] += record_stats_local["no_dirs"]
        # record_stats["no_files"] += record_stats_local["no_files"]
        pass
    return record_stats_local


def get_overall_stats_from_input_dirs(conf_dict, verbose = 0) -> (collections.OrderedDict, collections.namedtuple):
    
    keys = ["no_dirs", "no_configs_dirs", "no_trains_dirs"]
    keys += ["no_files", "no_models_files", "no_models_to_be_tested"]
    keys += ["no_finals_models", "no_pruned_models"]
    RecordStats = collections.namedtuple("RecordStats", keys)
    
    record_stats = collections.OrderedDict(zip(keys, [0] * len(keys)))
    for a_dir in conf_dict['input_dirs_list']:
        traverse_directory(a_dir, record_stats, keys)
        pass
    metadata_tbl = dict(
        tabular_data=[list(record_stats.values())],
        headers=list(record_stats.keys()),
    )
    
    if verbose == 1:
        table_stats = tabulate.tabulate(**metadata_tbl)
        print(table_stats)
    record_stats_nt = RecordStats._make(record_stats.values())
    return record_stats, record_stats_nt
