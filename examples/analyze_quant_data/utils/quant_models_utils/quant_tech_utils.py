from utils.libs.std_python_libs import *
from utils.libs.data_science_libs import *
from utils.libs.graphics_and_interactive_libs import *

BASELINE_PATH = "/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/train_uniform_baselines.csv"

CMPRSS_PATH = '/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/cmprs_df.csv'

ROOT_DIR = '/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/tmp_quantized_datasets'

UNWANTED_COLS = "Unnamed: 0,Unnamed: 0.1".split(",")
OLD_COLUMNS = "date_train,init_from,size_byte,footprint".split(",")
NEW_COLUMNS = "date,init-from,size(byte),footprint(%)".split(",")

# ================================================================================ #
# Function `load_full_cmprss_dataset`
# ================================================================================ #
def get_rid_of_unwanted_columns(a_df: pd.DataFrame) -> pd.DataFrame:
    for a_col in UNWANTED_COLS:
        if f"{a_col}" in a_df.columns:
            a_df = a_df.drop([f"{a_col}"], axis = 1)
            pass
        pass
    return a_df


def load_full_cmprss_dataset():
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


def add_missing_columns_as_targets(a_df: pd.DataFrame) -> pd.DataFrame:

    def create_prune_tech_attr(item):
        if '+' not in item: return "QUANT"
        return item.split("+")[0]
    def create_prune_quant_attr(item):
        if '+' not in item: return item
        return item.split("+")[1]
    def create_cmprss_class_2(item):
        if '+' not in item: return item
        return item.split("+")[1]
    def create_cmprss_class_3(item):
        if '+' not in item: return item
        return item.split("+")[1]
    def create_prune_rate_intervals(prune_rate):
        pos = np.arange(0, int(100 - prune_rate) + 5, 5)[-1]
        if pos == 0: return f"0"
        return f"{pos-5}-{pos}"
    
    if "prune_techs" not in a_df.columns:
        a_df["prune_techs"] = list(map(create_prune_tech_attr, a_df["cmprss-class"].values))
        pass
    if "prune_techs" not in a_df.columns:
        a_df["quant_techs"] = list(map(create_prune_quant_attr, a_df["cmprss-class"].values))
        pass
    if "prune_rate_intervals" not in a_df.columns:
        a_df["prune_rate_intervals"] = list(map(create_prune_rate_intervals, a_df["footprint(%)"].values))
        pass
    if "cmprss-class-2" not in a_df.columns:
        a_df["cmprss-class-2"] = list(map(create_cmprss_class_2, a_df["cmprss-class"].values))
        pass
    if "cmprss-class-3" not in a_df.columns:
        a_df["cmprss-class-3"] = list(map(create_cmprss_class_3, a_df["cmprss-class"].values))
        pass
    return a_df


def get_selected_dataset(a_dataset_path: str) -> pd.DataFrame:
    tmp_df = pd.read_csv(a_dataset_path)
    tmp_df = get_rid_of_unwanted_columns(tmp_df)
    tmp_df = rename_columns_and_get_rid_of_old_ones(tmp_df, OLD_COLUMNS, NEW_COLUMNS)
    tmp_df = add_missing_columns_as_targets(tmp_df)
    return tmp_df


# ================================================================================ #
# Function `get_list_of_quant_datasets`
# ================================================================================ #
def get_list_of_quanted_datasets(root_dir:str=ROOT_DIR):
    """
    TODO comment it
    """
    all_quanted_csv_list = []
    quanted_pairs_list = []
    quanted_csv_dict = dict()
    def filter_csv(item, root_dir=root_dir):
        if os.path.isfile(os.path.join(root_dir, item)):
            if os.path.exists(item) and os.path.isfile(item):
                if item.endswith(".csv"):
                    return True
        return False
    for dir_name, subdirs_list, files_list in os.walk(root_dir):
        files_list_full = list(map(lambda item: os.path.join(dir_name, item), files_list))
        quanted_csv_dict = list(filter(filter_csv, files_list_full))
        quanted_pairs_list.extend(
            list(map(lambda item: (os.path.basename(item), item), quanted_csv_dict))
        )
        all_quanted_csv_list.extend(list(map(lambda item: os.path.basename(item), files_list_full)))
        pass
    quanted_csv_dict = dict(quanted_pairs_list)
    return all_quanted_csv_list, quanted_csv_dict
