import argparse


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
        
    return parser
