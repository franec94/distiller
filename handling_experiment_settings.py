# ---------------------------------------------- #
# Python's Std, Community Packages
# ---------------------------------------------- #
from src.libraries.all_libs import *

# ---------------------------------------------- #
# Project Custom functions
# ---------------------------------------------- #
from src.generics.custom_cmd_line_parsers.create_experiments_custom_argparse import get_argparser
from src.generics.utils import read_conf_file_content

from src.create_experiments_settings.create_experiment_settings_utils import get_workload_infos
from src.create_experiments_settings.create_experiment_settings_utils import update_bp_confs
from src.create_experiments_settings.create_experiment_settings_utils import create_dataset_experiments
from src.create_experiments_settings.create_experiment_settings_utils import run_dataset_experiments


def main(args) -> None:
    # Init variables.
    out_dict_info: dict = None
    a_df: pd.DataFrame = pd.DataFrame()

    # Load Script conf dictionary, with options and
    # constraints by means leading workload to be carryed out.
    conf_dict: dict = read_conf_file_content(args.conf_file)
    # print(type(conf_dict))
    # pprint(conf_dict)
    assert type(conf_dict) == dict

    # Load scheduler scheme to be employed
    # as blueprint for crafting new ones.
    bp_conf_file_path: str = os.path.join(
        conf_dict["dataset"]["blueprint_conf_dirname"], conf_dict["dataset"]["blueprint_conf_filename"])
    bp_conf_dict: dict = read_conf_file_content(bp_conf_file_path)
    # print(type(bp_conf_dict))
    # pprint(bp_conf_dict)
    assert type(bp_conf_dict) == dict

    if args.summary_estimated_workload:
        get_workload_infos(conf_dict, bp_conf_dict)
        return
    if conf_dict["actions"]["create_dataset_exp"]:
        out_conf_list = \
            update_bp_confs(conf_dict, bp_conf_dict)
        a_df, out_dict_info = create_dataset_experiments(args, conf_dict, out_conf_list, echo=True, verbose=1)
        pass
    if conf_dict["actions"]["run_dataset_exp"]:
        out_dict_info = run_dataset_experiments(args, conf_dict, a_df, out_dict_info=out_dict_info)
        pass

    if out_dict_info:
        meta_tb = dict(
            tabular_data=out_dict_info.items()
        )
        table = tabulate.tabulate(**meta_tb)
        print(table)
        pass
    pass


if __name__ == "__main__":
    """
    try:
        parser = get_argparser()
        args = parser.parse_args()
        main(args)
    except Exception as err:
        print(f"{str(err)}")
        pass
    """
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
    pass