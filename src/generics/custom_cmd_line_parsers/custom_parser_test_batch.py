# from src.libraries.all_libs import *
import argparse


def get_custom_parser_test_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", dest="input_dataset_file", type=str, required=True,\
        help="Input file path, within local file system to dataset with configuration to be tested."
    )
    parser.add_argument("--output_dataset_path", dest="output_dataset_path", type=str, required=True,\
        help="Input dir path, within local file system to save output dataset."
    )
    parser.add_argument("--tests_logging_root", dest="tests_logging_root", type=str, required=True,\
        help="Input dir path, within local file system to save output tests results."
    )
    return parser
