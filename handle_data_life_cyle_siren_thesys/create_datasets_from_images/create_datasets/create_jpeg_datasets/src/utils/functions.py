from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *


def  get_root_level_logger(root_path, loggger_name='train.log'):
    """Get root level logger."""
    log_filename = os.path.join(root_path, f'{loggger_name}')
    log_filename_exists = check_file_exists(log_filename, raise_exception=False)
    if log_filename_exists:
        os.remove(log_filename)
        pass

    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
    pass


def check_file_exists(file_path: str, raise_exception:bool=True) -> bool:
    """Check whether input provided file path really exists.
    Args:
    -----
    `file_path` - str object related to input file path to be tested.\n
    `raise_exception` - bool object, set it to raise exception when tested input file does not exists indeed.\n
    Return:
    -------
    `bool` true if file exists otherwise false.\n
    """
    if not os.path.isfile(file_path):
        if raise_exception:
            raise Exception(f"Error: file '{file_path}' does not exists!")
        return False
    return True


def check_dir_exists(dir_path, raise_exception=True):
    """Check whether input provided file path really exists.
    Args:
    -----
    `dir_path` - str object related to input dir path to be tested.\n
    `raise_exception` - bool object, set it to raise exception when tested input dir does not exists indeed.\n
    Return:
    -------
    `bool` true if dir exists otherwise false.\n
    """
    if not os.path.isdir(dir_path):
        if raise_exception:
            raise Exception(f"Error: directory '{dir_path}' does not exists!")
        else: return False
    return True


def get_all_files_by_ext(dir_path, ext, recursive_search = False, regex_filter = None, verbose = 0):
    files_list = []
    check_dir_exists(dir_path)
    if isinstance(ext, list) == False:
        ext = [ext]
    if recursive_search:
        for a_ext in ext:
            data = pathlib.Path(f'{dir_path}').rglob(f'*.{a_ext}')
            if data != None:
                files_list.extend(data)
    else:
        for a_ext in ext:
            data = pathlib.Path(f'{dir_path}').glob(f'*.{a_ext}')
            if data != None:
                files_list.extend(data)
                if verbose == 1:
                    data_2 = pathlib.Path(f'{dir_path}').glob(f'*.{a_ext}')
                    for path in data_2:
                        print(path.name)
                files_list.extend(data)
    if len(files_list) == 0: return []
    
    def filter_files(a_file, raise_exception = False):
        return check_file_exists(a_file, raise_exception)
    files_list = list(filter(filter_files, files_list))
    
    if regex_filter != None:
        if isinstance(regex_filter, list) == False:
            regex_filter = [regex_filter]
            filtered_files = []
            for a_regex in regex_filter:
                res_tmp = list(filter(lambda xx: a_regex.match(xx.name) != None, files_list))
                filtered_files.extend(res_tmp)
        return filtered_files
    
    return files_list


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except PermissionError as err:
            print(f"Error raised when dealing with dir '{dir_path}' creation!", file=sys.stderr)
            print(str(err))
            sys.exit(-1)
            pass
        except:
            pass
        pass
    pass


def read_conf_file(conf_file_path, raise_exception = True):
    _ = check_file_exists(file_path=conf_file_path, raise_exception=raise_exception)
    try:
        with open(conf_file_path, "r") as f:
            conf_data = yaml.load(f, Loader=yaml.FullLoader)
            pass
        return conf_data
    except Exception as err:
        raise Exception(f"Error: when reading input conf file '{conf_file_path}'")
    pass


def load_target_image(image_file_path = None) -> PIL.Image:
    """TODO Comment it.
    Returns:
    --------
    `PIL.Image`
    """
    # image_file_path = 'test068.png'
    im = None
    if image_file_path != None:
        image_exists = check_file_exists(image_file_path, raise_exception=False)
        if image_exists:
            im = PIL.Image.open(f'{image_file_path}')
            return im
    
    im = PIL.Image.fromarray(skimage.data.camera())
    return im


def get_dataframe(conf_data):
    if 'result_timestamp' in conf_data.keys():
        result_timestamp = conf_data['result_timestamp']
        if result_timestamp == 'None' or result_timestamp is None:
            columns = conf_data['columns_df_str'].split(";")
            a_file = conf_data['result_file_path']
    
            check_file_exists(a_file, raise_exception=True)
            train_arr = np.loadtxt(a_file)
            train_df = pd.DataFrame(data = train_arr, columns = columns)
            return train_df
        
    index_timestamp = conf_data['results_timestamps'].index(result_timestamp)
    a_file, a_ts = \
        conf_data['results_file_paths'][index_timestamp], conf_data['results_timestamps'][index_timestamp]
    check_file_exists(a_file, raise_exception=True)
    train_arr = np.loadtxt(a_file)
    train_df = pd.DataFrame(data = train_arr, columns = columns)
    return train_df


def get_dict_dataframes(conf_data):
    columns = conf_data['columns_df_str'].split(";")
    result_dict_df = dict()
    for a_file, a_ts in zip(conf_data['results_file_paths'], conf_data['results_timestamps']):
        try:
            check_file_exists(a_file, raise_exception=True)
            train_arr = np.loadtxt(a_file)
            indeces = [a_ts] * len(train_arr)
            train_df = pd.DataFrame(data = train_arr, columns = columns, index=indeces)
            result_dict_df[a_ts] = train_df
        except Exception as _:
            pass
        pass
    return result_dict_df


def laod_data_from_files_list(files_list, as_df = False, columns = None):
    if files_list == None or len(files_list) == 0: return None
    data_arr = []
    def load_data_and_concat(a, b):
        b_arr = np.loadtxt(f"{b}")
        if len(a) == 0:
            return b_arr
        return np.concatenate((a, b_arr), axis=0)
    res_arr = functools.reduce(lambda a,b : load_data_and_concat(a, b), files_list, data_arr)
    if as_df is True:
        if columns != None:
            return pd.DataFrame(res_arr, columns = columns)
        else:
            return pd.DataFrame(res_arr)
    return res_arr


def get_image_details_as_table(image: PIL.Image):
    data_table = dict(
        name="Camera",
        shape=image.size,
        size_byte=image.size[0]*image.size[1],
        image_band=image.getbands(),
        entropy=image.entropy(),
    )
    meta_data_table = dict(
        tabular_data=data_table.items(),
        tablefmt="pipe" # "github"
    )
    table = tabulate.tabulate(**meta_data_table)
    # print(table)
    return table


def get_new_targets(target, size):
    offset = target // 2
    if target % 2 == 0:
        extreme_1 = size // 2
        residual = 0
    else:
        extreme_1 = size // 2 - 1
        residual = 1
        pass
    extreme_2 = size // 2
    return extreme_1 - offset + residual, extreme_2 + offset + residual


def get_cropped_by_center_image(im, target = 256) -> PIL.Image:
    """TODO comment it.
    Returns:
    --------
    `PIL.Image` - croppet image.\n
    """
    width, height = im.size

    if isinstance(target, int):
        target = (target, target)
        pass

    left, right = get_new_targets(target[0], width)
    top, bottom = get_new_targets(target[1], height)

    # print(im.crop((left, top, right, bottom)).size)
    # print((left, top, right, bottom))

    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped
