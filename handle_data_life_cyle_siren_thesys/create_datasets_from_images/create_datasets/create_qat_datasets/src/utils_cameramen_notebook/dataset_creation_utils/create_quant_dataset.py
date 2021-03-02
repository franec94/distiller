from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *
from src.data_loaders import dataset_loaders

import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from src.libs.project_libs import *
    pass

from src.utils_cameramen_notebook.utils_graphics import utils_graphics as cameramen_ugraph
from src.utils_cameramen_notebook.utils_graphics import create_plots as cameramen_cplots



def get_blueprint_options(image_name: str = "cameramen") -> (dict, object):
    """Get Blue Print for options.\n
    Returns:
    --------
    `conf_data_dict` - a dict .\n
    `table_conf_data` - tabulate.tabulate table.\n
    """

    ROOT_DIR = f"/media/franec94/Elements/Francesco/THESYS/FROM_DESKTOP/tmp_iside/tmp_qat/{image_name}/cameramen_1"
    OUT_DIR = "./out/results"
    CONF_FILEPATH = f"./confs/{image_name}/conf_{image_name}_init_from_pruned.yaml"
    TESTS_LOGGING_ROOTS = "./logs"
    OUTPUT_DATASET_PATH = "./out/datasets"
    if image_name == "test066":
        PERFORMANCES_PATH = f"/media/franec94/Elements/Francesco/THESYS/FROM_DESKTOP/tmp_iside/tmp_qat/{image_name}_airplane/performances.txt"
    else:
        PERFORMANCES_PATH = f"/media/franec94/Elements/Francesco/THESYS/FROM_DESKTOP/tmp_iside/tmp_qat/{image_name}/performances.txt"
        pass
    OUT_FILENAME = "out.csv"

    conf_data_dict = dict(
        root_dir=ROOT_DIR,
        out_dir=OUT_DIR,
        conf_filepath=CONF_FILEPATH,
        tests_logging_root=TESTS_LOGGING_ROOTS,
        output_dataset_path=OUTPUT_DATASET_PATH,
        performances_path=PERFORMANCES_PATH,
        # out_filename=OUT_FILENAME
    )
    # pprint(conf_data_dict)
    meta_data_table = dict(
        tabular_data=conf_data_dict.items()
    )
    table_conf_data = tabulate.tabulate(**meta_data_table)
    return conf_data_dict, table_conf_data


def create_quant_dataset(root_dirs_list:list = [], image_name:str = "cameramen", verbose:int = 0) -> pd.DataFrame:
    """Create quant dataset from list of dirs.\n
    Returns:
    --------
    `a_dataset_df` - pd.DataFrame.\n
    """

    empty_dataframe = pd.DataFrame()
    if not root_dirs_list or root_dirs_list == []:
        return empty_dataframe
    
    conf_data_dict, table_conf_data = \
        get_blueprint_options(image_name=image_name)
    if verbose > 0:
        print(table_conf_data)
    dfs_list: list = []

    total = len(root_dirs_list)
    with tqdm.tqdm_notebook(total=total) as pbar:
        for ii, ROOT_DIR in enumerate(root_dirs_list):
            pbar.write(f"Processing {ROOT_DIR}...")
            try:
                ROOT_DIR=ROOT_DIR
                OUT_DIR=conf_data_dict["out_dir"]
                CONF_FILEPATH=conf_data_dict["conf_filepath"]
                TESTS_LOGGING_ROOTS=conf_data_dict["tests_logging_root"]
                OUTPUT_DATASET_PATH=conf_data_dict["output_dataset_path"]
                PERFORMANCES_PATH=conf_data_dict["performances_path"]
                # OUT_FILENAME=conf_data_dict["out_filename"]

                OUT_FILENAME = "out.csv"

                conf_data_dict = dict(
                    root_dir=ROOT_DIR,
                    out_dir=OUT_DIR,
                    conf_filepath=CONF_FILEPATH,
                    tests_logging_root=TESTS_LOGGING_ROOTS,
                    output_dataset_path=OUTPUT_DATASET_PATH,
                    performances_path=PERFORMANCES_PATH
                )
                parser = cpqd.get_custom_parser_for_notebook(conf_data_dict=conf_data_dict)
                args, _  = parser.parse_known_args()
                conf_data: dict = read_conf_file(conf_file_path=args.conf_filepath)
                a_df = cqd.create_out_dataset(args=args, conf_data=conf_data, out_filename=OUT_FILENAME)
                merged_df = merge_performace_w_models_data(args=args, models_df=a_df)
                dfs_list.append(copy.deepcopy(merged_df))
            except Exception as err:
                # print("[*] Error occuring for:")
                # print(f"\t{ii} - {ROOT_DIR}")
                # print(f"\tError: {str(err)}")
                pbar.write("[*] Error occuring for:")
                pbar.write(f"\t{ii} - {ROOT_DIR}")
                pbar.write(f"\tError: {str(err)}")
                pass
            pbar.update(1)
            pass
    if dfs_list == []: return empty_dataframe
    quant_df = pd.concat(dfs_list, axis = 0, ignore_index=True)
    return quant_df
