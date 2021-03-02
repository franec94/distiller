from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from src.handle_raw_data.load_data_from_conf import load_rawdata_from_conf
from src.end_to_end_utils.end_to_end_utils import keep_target_qualities
from src.utils.functions import load_target_image, get_cropped_by_center_image

# from src.libs.all_end_to_end_exp_analyses import *
from src.end_to_end_utils.create_out_dirs import merge_datasets_quant_data, get_some_dfs
import PIL


def check_image_name_in_bsd_dataset(image_name:str):
    a_image = None
    root_dir_bsd68: str = "/home/franec94/Documents/thesys-siren/data/testsets/BSD68"
    onlyfiles = [f for f in os.listdir(root_dir_bsd68) if os.path.isfile(os.path.join(root_dir_bsd68, f))]

    # image_path = os.path.join(root_dir_bsd68, image_name)
    for a_tmp_image in onlyfiles:
        a_tmp_image = os.path.join(root_dir_bsd68, a_tmp_image)
        a_tmp_image_name = os.path.basename(a_tmp_image)
        a_tmp_image_name, _ = os.path.splitext(a_tmp_image_name)
        if a_tmp_image_name == image_name:
            a_image = PIL.Image.open(a_tmp_image)
            return a_image
        pass
    return a_image


def load_image_by_name(image_name:str, cropped_center=None) -> PIL.Image:
    a_image = check_image_name_in_bsd_dataset(image_name=image_name)
    if a_image:
        if cropped_center:
            a_image = get_cropped_by_center_image(im=a_image, target=cropped_center)
        return a_image
    elif image_name == "cameramen":
        a_image = PIL.Image.fromarray(skimage.data.camera())
        if cropped_center:
            a_image = get_cropped_by_center_image(im=a_image, target=cropped_center)
        return a_image
    else:
        raise Exception(f"Provided image name: {image_name} not allowed!")


def load_target_images(image_file_path: str = None, target=256) -> (PIL.Image, PIL.Image, int):
    """TODO comment .it
    Returns:
    -------
    `PIL.Image` - full image .\n
    `PIL.Image` -  cropped image.\n
    `int` - byte sieze full image .\n.\n
    `int` - byte size cropped image.\n.\n
    """
    camera = load_target_image(image_file_path=image_file_path)
    camera_crop = get_cropped_by_center_image(im=camera, target=target)
    size_byte_full = sys.getsizeof(camera.tobytes())
    size_byte_crop = sys.getsizeof(camera_crop.tobytes())
    return camera, camera_crop, size_byte_full, size_byte_crop

# ---------------------------------------------------------------- #
# Minor Datasets
# ---------------------------------------------------------------- #

def get_teacher_model(baseline_df: pd.DataFrame, unique_pairs: list, n_hf_t: int, n_hl_t: int) -> pd.DataFrame:
    """TODO comment .it"""
    targtes_dfs_list: list = []
    for a_pair in unique_pairs:
        n_hf, n_hl = a_pair
        if n_hf != n_hf_t: continue
        if n_hl != n_hl_t: continue

        pos = baseline_df["cmprss-class"].str.contains(f'{n_hf}')
        pos2 = baseline_df["cmprss-class"].str.contains(f'{n_hl}')
        a_sub_df = baseline_df[(pos) & (pos2)]
        if a_sub_df.shape[0] != 0:
            targtes_dfs_list.append(a_sub_df.head(1))
        pass
    if targtes_dfs_list == []: return pd.DataFrame()
    teachers_bslns_df = pd.concat(targtes_dfs_list, axis = 0, ignore_index = True)

    return teachers_bslns_df


def load_tch_dataset(conf_data: dict, a_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Comment it.
    Returns:
    --------
    `pd.DataFrame` - required dataset.\n
    """
    unique_pairs = set(map(lambda item: (item[0], item[1]), list(a_df[["n_hf", "n_hl"]].values)))

    # n_hf_t = conf_data["raw_data"]["model_infos"]["n_hf"]
    # n_hl_t = conf_data["raw_data"]["model_infos"]["n_hl"]
    n_hf_t = conf_data["n_hf"]
    n_hl_t = conf_data["n_hl"]

    tch_df = get_teacher_model(baseline_df, unique_pairs, n_hf_t, n_hl_t)
    # tch_df

    return tch_df


def get_agp_pruned_model(pruned_df: pd.DataFrame, conf_data_dict: dict) -> pd.DataFrame:

    timestamp: str = conf_data_dict["timestamp"]
    
    pos = pruned_df["date"] == timestamp
    pruned_model_df: pd.DataFrame = pruned_df[pos].head(1)
    return pruned_model_df


def get_pruned_df_data(conf_data: dict, a_df: pd.DataFrame, agp_df: pd.DataFrame) -> pd.DataFrame:
    """Comment it.
    Returns:
    --------
    `pd.DataFrame` - required dataset.\n
    """
    # conf_data_dict: dict = conf_data["raw_data"]["filter_data"]["agp"]
    conf_data_dict = copy.deepcopy(conf_data)
    pruned_model_df = get_agp_pruned_model(pruned_df=agp_df, conf_data_dict=conf_data_dict)
    # pruned_model_df
    
    return pruned_model_df


def get_min_max_jpeg_df_data(conf_data: dict, jpeg_df: pd.DataFrame) -> pd.DataFrame:
    """Comment it.
    Returns:
    --------
    `pd.DataFrame` - required dataset.\n
    """
    conf_data_dict = conf_data["raw_data"]["filter_data"]["jpeg"]
    _, tmp_jpeg_max_q, tmp_jpeg_min_q = \
        keep_target_qualities(jpeg_df, jpeg_df, conf_data_dict,)
    return tmp_jpeg_max_q, tmp_jpeg_min_q

# ---------------------------------------------------------------- #
# Jpeg Data Loader
# ---------------------------------------------------------------- #

def load_jpeg_dataset_as_bunch_container() -> sklearn.utils.Bunch:
    jpeg_df, _, _, _ = get_some_dfs()
    jpeg_dataset = sklearn.utils.Bunch(
        data=jpeg_df.drop(["psnr"], axis = 1), target=jpeg_df["psnr"])
    return jpeg_dataset


def load_jpeg_dataset_as_dataframe_container(image_name="cameramen") -> pd.DataFrame:
    datasets_dirs: dict = dict(
        cameramen="/home/franec94/Documents/thesys-siren/data/datasets/cameramen",
        test066="/home/franec94/Documents/thesys-siren/data/datasets/test066_datasets",
        test068="/home/franec94/Documents/thesys-siren/data/datasets/test068_datasets"
    )

    """if image_name.lower() == "cameramen":
        jpeg_df, _, _, _= get_some_dfs()
        return jpeg_df
    """
    if image_name.lower() in datasets_dirs.keys():
        jpeg_dataset_path = os.path.join(datasets_dirs[image_name.lower()], "jpeg_dataset.csv")
        jpeg_df = pd.read_csv(jpeg_dataset_path)
        return jpeg_df
    else:
        raise Exception(f"Error: {image_name} provided is not allowed!")    
    pass


def load_jpeg_dataset(dtype: str = "bunch", image_name = "cameramen") -> object:
    """Load Jpeg dataset.
    Returns:
    --------
    `jpeg_dataset` - either Bunch or Dataset like objects.\n
    """
    dtypes_list: list = list(map(str.lower, "bunch,dataframe".split(",")))
    dtypes_set: tuple = tuple(dtypes_list)
    if dtype.lower() not in dtypes_set:
        raise Exception(f"Error: provided dtype '{dtype}' not allowed!")
    if dtype.lower() == "bunch".lower():
        jpeg_dataset = load_jpeg_dataset_as_bunch_container()
    if dtype.lower() == "dataframe".lower():
        # raise Exception(f"Warning: provided dtype '{dtype}' not yet implemented!")
        jpeg_dataset = load_jpeg_dataset_as_dataframe_container(image_name=image_name)
    return jpeg_dataset


# ---------------------------------------------------------------- #
# AGP Data Loader
# ---------------------------------------------------------------- #

def load_pruning_dataset_as_bunch_container() -> sklearn.utils.Bunch:
    _, _, _, pruning_df = get_some_dfs()
    pruning_dataset = sklearn.utils.Bunch(
        data=pruning_df.drop(["psnr"], axis = 1), target=pruning_df["psnr"])
    return pruning_dataset


def load_pruning_dataset_as_dataframe_container(image_name: str = "cameramen") -> pd.DataFrame:
    datasets_dirs: dict = dict(
        test066="/home/franec94/Documents/thesys-siren/data/datasets/test066_datasets",
        test068="/home/franec94/Documents/thesys-siren/data/datasets/test068_datasets"
    )
    if image_name.lower() == "cameramen":
        _, _, _, pruning_df = get_some_dfs()
        return pruning_df
    elif image_name.lower() in datasets_dirs.keys():
        jpeg_dataset_path = os.path.join(datasets_dirs[image_name.lower()], "pruned_dataset.csv")
        pruning_df = pd.read_csv(jpeg_dataset_path)
        return pruning_df
    else:
        raise Exception(f"Error: {image_name} provided is not allowed!")   
    pass


def load_prunining_dataset(dtype: str = "bunch", image_name: str = "cameramen") -> object:
    """Load Jpeg dataset.
    Returns:
    --------
    `pruning_dataset` - either Bunch or Dataset like objects.\n
    """
    dtypes_list: list = list(map(str.lower, "bunch,dataframe".split(",")))
    dtypes_set: tuple = tuple(dtypes_list)
    if dtype.lower() not in dtypes_set:
        raise Exception(f"Error: provided dtype '{dtype}' not allowed!")
    if dtype.lower() == "bunch".lower():
        pruning_dataset = load_pruning_dataset_as_bunch_container()
    if dtype.lower() == "dataframe".lower():
        # raise Exception(f"Warning: provided dtype '{dtype}' not yet implemented!")
        pruning_dataset = load_pruning_dataset_as_dataframe_container(image_name=image_name)
    return pruning_dataset


# ---------------------------------------------------------------- #
# Baseline Data Loader
# ---------------------------------------------------------------- #

def load_siren_baselines_dataset_as_bunch_container() -> sklearn.utils.Bunch:
    _,  siren_bsln_df, _,_ = get_some_dfs() #  siren_bsln_df, baseline_df
    siren_bsln_dataset = sklearn.utils.Bunch(
        data=siren_bsln_df.drop(["psnr"], axis = 1), target=siren_bsln_df["psnr"])
    return siren_bsln_dataset


def load_siren_baselines_dataset_as_dataframe_container() -> pd.DataFrame:
    _,  siren_bsln_df, _, _ = get_some_dfs() #  siren_bsln_df, baseline_df
    return siren_bsln_df


def load_siren_baselines_dataset(dtype: str = "bunch") -> object:
    """Load Jpeg dataset.
    Returns:
    --------
    `siren_bsln_df` - either Bunch or Dataset like objects.\n
    """
    dtypes_list: list = list(map(str.lower, "bunch,dataframe".split(",")))
    dtypes_set: tuple = tuple(dtypes_list)
    if dtype.lower() not in dtypes_set:
        raise Exception(f"Error: provided dtype '{dtype}' not allowed!")
    if dtype.lower() == "bunch".lower():
        siren_bsln_dataset = load_siren_baselines_dataset_as_bunch_container()
    if dtype.lower() == "dataframe".lower():
        # raise Exception(f"Warning: provided dtype '{dtype}' not yet implemented!")
        siren_bsln_dataset = load_siren_baselines_dataset_as_dataframe_container()
    return siren_bsln_dataset

# ---------------------------------------------------------------- #
# Quant Dataset
# ---------------------------------------------------------------- #

def load_quant_dataset_as_bunch_container_v2(conf_data: dict) -> sklearn.utils.Bunch:
    conf_data 
    (camera, camera_crop, size_byte_crop), a_df, (success_readings, failure_readings) = \
        load_rawdata_from_conf(conf_data=conf_data, verbose=1)
    return a_df


def load_quant_dataset_as_dataframe_container_v2(conf_data: dict) -> pd.DataFrame:
    conf_data 
    (camera, camera_crop, size_byte_crop), a_df, (success_readings, failure_readings) = \
        load_rawdata_from_conf(conf_data=conf_data, verbose=1)
    return a_df


def load_quant_dataset_v2(conf_data: dict, dtype: str = "bunch") -> object:
    """Load Jpeg dataset.
    Returns:
    --------
    `quant_dataset` - either Bunch or Dataset like objects.\n
    """
    dtypes_list: list = list(map(str.lower, "bunch,dataframe".split(",")))
    dtypes_set: tuple = tuple(dtypes_list)
    if dtype.lower() not in dtypes_set:
        raise Exception(f"Error: provided dtype '{dtype}' not allowed!")
    if dtype.lower() == "bunch".lower():
        quant_dataset = load_quant_dataset_as_bunch_container_v2(conf_data=conf_data)
    if dtype.lower() == "dataframe".lower():
        # raise Exception(f"Warning: provided dtype '{dtype}' not yet implemented!")
        quant_dataset = load_quant_dataset_as_dataframe_container_v2(conf_data=conf_data)
    return quant_dataset


ROOT_QUANT_DATASET = "/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/datasets/quant_dataset"

def load_quant_dataset_as_bunch_container(conf_data: dict = None, verbose: int = 0) -> sklearn.utils.Bunch:
    dfs_list: list = []
    search_in = f'*.csv'
    total = len(list(Path(f'{ROOT_QUANT_DATASET}').rglob(search_in)))
    if verbose == 1:
        with tqdm.tqdm(total=total) as pbar:
            for a_path in Path(f'{ROOT_QUANT_DATASET}').rglob(search_in):
                pbar.write(a_path.name)
                dfs_list.append(pd.read_csv(a_path))
                pass
            pbar.update(1)
            pass
    else:
        for a_path in Path(f'{ROOT_QUANT_DATASET}').rglob(search_in):
            dfs_list.append(pd.read_csv(a_path))
            pass
        pass
    quant_df = pd.concat(dfs_list, axis = 0, ignore_index=True)
    quant_dataset = sklearn.utils.Bunch(data=quant_df.drop(["psnr"], axis=1), target=quant_df["psnr"])
    return quant_dataset


def load_quant_dataset_as_dataframe_container(conf_data: dict = None, verbose: int = 0) -> pd.DataFrame:
    dfs_list: list = []
    search_in = f'*.csv'
    total = len(list(Path(f'{ROOT_QUANT_DATASET}').rglob(search_in)))
    
    # print(total); sys.exit(-1)

    if not os.path.exists(ROOT_QUANT_DATASET): print(f"{ROOT_QUANT_DATASET} not exists!"); sys.exit(-1)
    if not os.path.isdir(ROOT_QUANT_DATASET): print(f"{ROOT_QUANT_DATASET} is not a dir!"); sys.exit(-1)

    if verbose == 1:
        with tqdm.tqdm(total=total) as pbar:
            for a_path in Path(f'{ROOT_QUANT_DATASET}').rglob(search_in):
                pbar.write(a_path.name)
                dfs_list.append(pd.read_csv(a_path))
                pass
            pbar.update(1)
            pass
    else:
        for a_path in Path(f'{ROOT_QUANT_DATASET}').rglob(search_in):
            dfs_list.append(pd.read_csv(a_path))
            pass
        pass
    quant_df = pd.concat(dfs_list, axis = 0, ignore_index=True)
    return quant_df


def load_quant_dataset(conf_data: dict = None, dtype: str = "bunch", verbose:int=0) -> object:
    """Load Jpeg dataset.
    Returns:
    --------
    `quant_dataset` - either Bunch or Dataset like objects.\n
    """
    dtypes_list: list = list(map(str.lower, "bunch,dataframe".split(",")))
    dtypes_set: tuple = tuple(dtypes_list)
    if dtype.lower() not in dtypes_set:
        raise Exception(f"Error: provided dtype '{dtype}' not allowed!")
    if dtype.lower() == "bunch".lower():
        quant_dataset = load_quant_dataset_as_bunch_container(conf_data=conf_data, verbose=verbose)
    if dtype.lower() == "dataframe".lower():
        # raise Exception(f"Warning: provided dtype '{dtype}' not yet implemented!")
        quant_dataset = load_quant_dataset_as_dataframe_container(conf_data=conf_data, verbose=verbose)
    return quant_dataset
