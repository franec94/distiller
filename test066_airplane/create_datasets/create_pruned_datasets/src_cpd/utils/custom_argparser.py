import argparse

def get_custom_parser_from_dict(conf_data_dict: dict) -> argparse.ArgumentParser:
    """Return custom argument parser.
    Returns:
    --------
    `parser` - argparse.ArgumentParser.\n
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, dest="root_dir", default=conf_data_dict["root_dir"], required=False, \
        help="Input path to local file system directory that represents root directory for overall trained models.")

    parser.add_argument("--out_dir", type=str, dest="out_dir", default=conf_data_dict["out_dir"], required=False, \
        help="Input path to output local file system directory that will be created for storing results.")

    parser.add_argument("--conf_filepath", type=str, dest="conf_filepath", default=conf_data_dict["conf_filepath"], required=False, \
        help="Input path to local file system file that will contain metada for correctly process data.")

    parser.add_argument("--tests_logging_root", type=str, dest="tests_logging_root", default=conf_data_dict["tests_logging_root"], required=False, \
        help="Output path to local file system directory that will contain results related to tested ahead of time pruned models.")

    parser.add_argument("--output_dataset_path", type=str, dest="output_dataset_path", default=conf_data_dict["output_dataset_path"], required=False, \
        help="Output path to local file system directory that will contain results related to tested ahead of time pruned models.")
    
    parser.add_argument("--performances_path", type=str, dest="performances_path", default=conf_data_dict["performances_path"], required=False, \
        help="Input path to local file system file that will contain results related to tested pruned models.")
    return parser



def get_custom_parser_for_notebook(conf_data_dict: dict) -> argparse.ArgumentParser:
    """Return custom argument parser.
    Returns:
    --------
    `parser` - argparse.ArgumentParser.\n
    """
    parser = get_custom_parser_from_dict(conf_data_dict=conf_data_dict)
    return parser


def get_custom_argparser() -> argparse.ArgumentParser:
    """Return custom argument parser.
    Returns:
    --------
    `parser` - argparse.ArgumentParser.\n
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, dest="root_dir", required=True, \
        help="Input path to local file system directory that represents root directory for overall trained models.")
    parser.add_argument("--out_dir", type=str, dest="out_dir", required=True, \
        help="Input path to output local file system directory that will be created for storing results.")
    parser.add_argument("--conf_filepath", type=str, dest="conf_filepath", required=True, \
        help="Input path to local file system file that will contain metada for correctly process data.")

    parser.add_argument("--tests_logging_root", type=str, dest="tests_logging_root", required=True, \
        help="Output path to local file system directory that will contain results related to tested ahead of time pruned models.")
    parser.add_argument("--output_dataset_path", type=str, dest="output_dataset_path", required=True, \
        help="Output path to local file system directory that will contain results related to tested ahead of time pruned models.")
    
    parser.add_argument("--performances_path", type=str, dest="performances_path", required=True, \
        help="Input path to local file system file that will contain results related to tested pruned models.")
    return parser
