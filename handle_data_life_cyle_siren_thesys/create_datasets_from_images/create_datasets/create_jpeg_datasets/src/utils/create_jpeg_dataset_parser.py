import argparse


def get_create_jpeg_dataset_parser() -> argparse.ArgumentParser:
    """Get `argparse.ArgumentParser` instance for script that is in charge
    of creating output dataset from configuration file specifics.\n
    Returns:
    --------
    `parser` - argparse.ArgumentParser.\n
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file",
        dest="conf_file", default=None,
        type=str, required=False,
        help="Configuration file path, within local file system."
    )
    parser.add_argument("--input_image",
        dest="input_image", default=None,
        type=str, required=False,
        help="Input image from which dataset will be created."
    )
    parser.add_argument("--output_dir",
        dest="output_dir", default=".",
        type=str, required=False,
        help="Output directory location where results will be stored."
    )
    return parser