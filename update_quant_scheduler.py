from pprint import pprint
import os
import sys
import pandas as pd
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--compress", type=str, required=True, dest="compress", \
    help="Input file path, within local file system, to resource that should be updated and contains constranints to be used for quant compression process of a DNN model."
)
parser.add_argument("--combs", type=str, required=True, dest="combs", \
    help="Input file path, within local file system, to resource that contains a set of possible options for updating scheduler yaml-compliant file."
)
parser.add_argument("--pos_comb", type=int, required=True, dest="pos_comb", \
    help="Input pos to comn to be exploited."
)


def check_file_exists(file_path: str, ext: str=None) -> None:
    """Check wheter input file path exists.
    Args:
    -----
    `file_path` - str python object that addresses input file path within local file system.\n
    `ext` - str python object that represent file extension to be checked if it is provided, so if it is not None (default=None).\n
    Return:
    ------
    `dict`-  python dict object with details of how to carry out compression proces by means of quantization trick.\n
    """
    if os.path.exists(file_path) is False:
        print(f"Error: resource '{file_path}' not exists!", file=sys.stderr)
        sys.exit(-1)
    if os.path.isfile(file_path) is False:
        print(f"Error: resource '{file_path}' is not a file!", file=sys.stderr)
        sys.exit(-2)
    if ext:
        _, file_ext = os.path.splitext(file_path)
        if ext != file_ext:
            print(f"Error: resource '{file_path}' is not a {ext} file!", file=sys.stderr)
            sys.exit(-3)
    pass


def read_yaml_file_to_dict(args) -> dict:
    """Read input file content which is a yaml file like.
    Args:
    -----
    `args` - Namespace object.\n
    Return:
    ------
    `dict`-  python dict object with details of how to carry out compression proces by means of quantization trick.\n
    """
    with open(args.compress) as compress_file:
        compress_dict = yaml.load(compress_file, Loader=yaml.FullLoader)
        # pprint(compress_dict)
    return compress_dict


def update_linear_quantizer(a_row: pd.DataFrame, compress_dict: dict, compress_file_path: str, pos_comb) -> None:
    """Update scheduler for linear quantization.
    Args:
    -----
    `a_row` - pd.DataFrame with details to be exploited for updating data dictionary representing quant options.\n
    `compress_dict` - dict python object representing quant options.\n
    `compress_file_path` - str python object, representing dest file path where updated options should be saved in.\n
    """
    compress_dict["quantizers"]["linear_quantizer"]["mode"] = a_row["mode"]
    compress_dict["quantizers"]["linear_quantizer"]["per_channel_wts"] = a_row["per_channel_wts"]

    compress_dict["policies"][0]["starting_epoch"] = a_row["starting_epoch"]
    compress_dict["policies"][0]["ending_epoch"] = a_row["ending_epoch"]
    compress_dict["policies"][0]["frequency"] = a_row["frequency"]

    pprint(compress_dict)
    
    base_name = os.path.basename(compress_file_path)
    dir_name = os.path.dirname(compress_file_path)

    new_filename = os.path.join(dir_name, f"comp_{pos_comb}_{base_name}")

    with open(f'{new_filename}', 'w') as outfile:
        yaml.dump(compress_dict, outfile, default_flow_style=False)
        pass
    pass


def main(args):
    """Entry point for doing updates required to modify scheduler yaml-like file.
    Args:
    -----
    `args` - Namespace object.\n
    """

    check_file_exists(args.compress, ".yaml")
    check_file_exists(args.combs, ".csv")

    df = pd.read_csv(args.combs)
    if "Unnmaded: 0" in df.columns:
        df = df.drop(["Unnmaded: 0"], axis = 1)

    if args.pos_comb < 0 or args.pos_comb > df.shape[0]:
        print(f"Error: pos='{args.pos_comb}' is not allowed since either negative or greater than {df.shape[0]}", file=sys.stderr)
        sys.exit(-1)

    a_row = dict(df.iloc[args.pos_comb, :])
    # pprint(a_row)
    compress_dict = read_yaml_file_to_dict(args)

    if "linear_quantizer" in compress_dict["quantizers"].keys():
        update_linear_quantizer(a_row, compress_dict, args.compress, args.pos_comb)
        pass
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass