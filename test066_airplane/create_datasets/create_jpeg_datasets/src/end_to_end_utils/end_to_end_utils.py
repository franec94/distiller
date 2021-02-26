from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

BASELINE_PATH = "/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/train_uniform_baselines.csv"

CMPRSS_PATH = '/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/cmprs_df.csv'

# ROOT_DIR = '/home/franec94/Documents/thesys-siren/codebase/results/tests/datasets'

UNWANTED_COLS = "Unnamed: 0,Unnamed: 0.1".split(",")
OLD_COLUMNS = "date_train,init_from,size_byte,footprint".split(",")
NEW_COLUMNS = "date,init-from,size(byte),footprint(%)".split(",")

# ================================================================================ #
# Function `load_full_cmprss_dataset`
# ================================================================================ #

def get_rid_of_unwanted_columns(a_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO - comment it: get_rid_of_unwanted_columns
    """
    for a_col in UNWANTED_COLS:
        if f"{a_col}" in a_df.columns:
            a_df = a_df.drop([f"{a_col}"], axis = 1)
            pass
        pass
    return a_df


def load_full_cmprss_dataset():
    """TODO comment .it"""
    baseline_df = pd.read_csv(f"{BASELINE_PATH}")
    cmprss_df = pd.read_csv(f"{CMPRSS_PATH}")
    
    baseline_df = get_rid_of_unwanted_columns(baseline_df)
    cmprss_df = get_rid_of_unwanted_columns(cmprss_df)

    def create_cmprss_class_3(item):
        a_class, a_class_2 = item
        if a_class_2 in "AGP,BASELINE,JPEG".split(","):
            if a_class_2 == 'BASELINE': return 'SIREN'
            return a_class_2
        return a_class
    def update_cmprss_class_2(item):
        a_class, a_class_2 = item
        if a_class_2 == 'BASELINE': return 'SIREN'
        return a_class_2
    def create_prune_rate_intervals(item):
        a_class, a_class_2, ftpr = item
        if a_class_2.upper() == "Agp".upper():
            pos = np.arange(0, int(100 - ftpr) + 5, 5)[-1]
            if pos == 0: return f"0"
            return f"{pos-5}-{pos}"
        return a_class_2
    
    cmprss_df["cmprss-class-3"] = list(map(create_cmprss_class_3, cmprss_df[["cmprss-class", "cmprss-class-2"]].values))
    
    cmprss_df["cmprss-class-2"] = list(map(update_cmprss_class_2, cmprss_df[["cmprss-class", "cmprss-class-2"]].values))
    baseline_df["cmprss-class-2"] = baseline_df["cmprss-class"] # list(map(update_cmprss_class_2, baseline_df[["cmprss-class", "cmprss-class-2"]].values))

    cmprss_df["prune_rate_intervals"] = list(map(create_prune_rate_intervals, cmprss_df[["cmprss-class", "cmprss-class-2", "footprint(%)"]].values))
    baseline_df["prune_rate_intervals"] = list(map(create_prune_rate_intervals, baseline_df[["cmprss-class", "cmprss-class-2", "footprint(%)"]].values))
    
    return baseline_df, cmprss_df

# ================================================================================ #
# Function `get_selected_dataset`
# ================================================================================ #

def rename_columns_and_get_rid_of_old_ones(a_df: pd.DataFrame, old_columns: list, new_columns: list) -> pd.DataFrame:
    for a_old_col, a_new_col in list(zip(old_columns, new_columns)):
        if a_old_col not in a_df.columns:
            continue
        a_df[f"{a_new_col}"] = a_df[f"{a_old_col}"]
        a_df = a_df.drop([f"{a_old_col}"], axis = 1)
        pass
    return a_df


def add_missing_columns_as_targets(a_df: pd.DataFrame, default_label='BASELINE') -> pd.DataFrame:
    if "prune_techs" not in a_df.columns:
        a_df["prune_techs"] = [f'{default_label}']* a_df.shape[0]
        pass
    if "prune_techs" not in a_df.columns:
        a_df["prune_techs"] = [0]* a_df.shape[0]
        pass
    if "prune_rate_intervals" not in a_df.columns:
        a_df["prune_techs"] = [f'{default_label}'] * a_df.shape[0]
        pass
    if "cmprss-class-2" not in a_df.columns:
        a_df["prune_techs"] = [f'{default_label}']* a_df.shape[0]
        pass
    if "cmprss-class-3" not in a_df.columns:
        a_df["prune_techs"] = [f'{default_label}']* a_df.shape[0]
        pass

    columns = list(a_df.columns)
    indeces = np.arange(0, len(columns))
    pos_key_dict = dict(zip(columns, indeces))
    def create_cmprss_class(item):
        n_hf, n_hl = item[pos_key_dict["n_hf"]], item[pos_key_dict["n_hl"]]
        epochs = item[pos_key_dict["num_epochs"]]
        cmprss_class = f'{default_label}_hf={n_hf:.0f}_hl={n_hl:.0f}_epochs={epochs:.0f}'
        return cmprss_class
    if "cmprss-class" not in a_df.columns:
        # a_df["cmprss-class"] = [f'{default_label}'] * a_df.shape[0]
        a_df["cmprss-class"] = list(map(create_cmprss_class, a_df.values))
        pass
    if "cmprss-class-2" not in a_df.columns:
        a_df["cmprss-class-2"] = [f'{default_label}'] * a_df.shape[0]
    if "cmprss-class-3" not in a_df.columns:
        a_df["cmprss-class-3"] = [f'{default_label}'] * a_df.shape[0]
    return a_df


def get_selected_dataset(a_dataset_path: str) -> pd.DataFrame:
    tmp_df = pd.read_csv(a_dataset_path)
    tmp_df = get_rid_of_unwanted_columns(tmp_df)
    tmp_df = rename_columns_and_get_rid_of_old_ones(tmp_df, OLD_COLUMNS, NEW_COLUMNS)
    tmp_df = add_missing_columns_as_targets(tmp_df)
    return tmp_df


def keep_target_qualities(jpeg_df, jpeg_filtered_df, conf_data_dict, a_key = 'psnr', columns = ["bpp", "psnr", "cmprss-class-2"]):
    """TODO Comment it."""

    min_q = conf_data_dict["min_q"]
    max_q = conf_data_dict["max_q"]

    cols = jpeg_df.columns
    def filter_by_quality_min(a_row, min_q = min_q, cols=cols):
        a_row = pd.Series(data=a_row, index=cols)
        q = float(a_row["cmprss-class"].split(":")[1])
        return q <= min_q
    pos_min = list(map(filter_by_quality_min, jpeg_df.values))
    
    def filter_by_quality_max(a_row, max_q = max_q, cols=cols):
        a_row = pd.Series(data=a_row, index=cols)
        q = float(a_row["cmprss-class"].split(":")[1])
        return q >= max_q
    pos_max = list(map(filter_by_quality_max, jpeg_df.values))

    cols = list(jpeg_df.columns)
    tmp_jpeg_min = jpeg_df[pos_min].sort_values(by=["psnr"], ascending=False).iloc[0,:]
    tmp_jpeg_min_q = pd.DataFrame([tmp_jpeg_min.values], columns = tmp_jpeg_min.index)
    # pprint(tmp_jpeg_min)

    tmp_jpeg_max = jpeg_df[pos_max].sort_values(by=["psnr"], ascending=False).iloc[0,:]
    tmp_jpeg_max_q = pd.DataFrame([tmp_jpeg_max.values], columns = tmp_jpeg_max.index)
    # pprint(tmp_jpeg_max)
    # sys.exit(0)

    tmp_jpeg = pd.concat([tmp_jpeg_min_q, tmp_jpeg_max_q], axis = 0,  ignore_index = True)
    tmp_jpeg.columns = cols
    tmp_jpeg = tmp_jpeg[columns]

    jpeg_filtered_df = pd.concat([jpeg_filtered_df, tmp_jpeg], axis = 0, ignore_index = True)
    return jpeg_filtered_df, tmp_jpeg_max_q, tmp_jpeg_min_q


def filter_dataframe_by_conf(a_df, conf_data_dict, a_key = 'psnr') -> pd.DataFrame:
    """TODO Comment it.
    Returns:
    --------
    `pd.DataFrame` - updated and filtered dataset.\n
    """
    
    sampling_datapoints = None
    
    min_psnr = conf_data_dict["min_psnr"]
    if not min_psnr:
        min_psnr = min(a_df[f"{a_key}"].values)
    max_psnr = conf_data_dict["max_psnr"]
    if not max_psnr:
        max_psnr = max(a_df[f"{a_key}"].values)
    if "sampling_datapoints" in conf_data_dict.keys():
        sampling_datapoints = conf_data_dict["sampling_datapoints"]

    pos = (a_df[f"{a_key}"] > min_psnr) & (a_df[f"{a_key}"] < max_psnr)

    if "min_bpp" in conf_data_dict.keys():
        min_bpp = conf_data_dict["min_bpp"]
        if min_bpp:
            pos = pos & (a_df[f"bpp"] > min_bpp)
    if "max_bpp" in conf_data_dict.keys():
        max_bpp = conf_data_dict["max_bpp"]
        if max_bpp:
            pos = pos & (a_df[f"bpp"] <= max_bpp)
        pass
    filtered_df = copy.deepcopy(a_df[pos])

    if sampling_datapoints:
        pos = np.arange(0, filtered_df.shape[0], sampling_datapoints)
        filtered_df = filtered_df.iloc[pos,:]
    return filtered_df


def filter_qatlrq_conf(a_df, conf_data_dict, jpeg_min_q_df, a_key = 'psnr'):
    """TODO Comment it."""
    
    filtered_df = None
    if not conf_data_dict or "delta_psnr"  not in conf_data_dict.keys(): return a_df

    delta_psnr = conf_data_dict["delta_psnr"]
    jmin_psnr = jpeg_min_q_df[f"{a_key}"].values[0]

    pos = a_df[f"{a_key}"] >= jmin_psnr - delta_psnr
    filtered_df = copy.deepcopy(a_df[pos])

    if "sampling_datapoints" in conf_data_dict.keys():
        sampling_datapoints = conf_data_dict["sampling_datapoints"]
    if sampling_datapoints:
        pos = np.arange(0, filtered_df.shape[0], sampling_datapoints)
        filtered_df = filtered_df.iloc[pos,:]
    return filtered_df
