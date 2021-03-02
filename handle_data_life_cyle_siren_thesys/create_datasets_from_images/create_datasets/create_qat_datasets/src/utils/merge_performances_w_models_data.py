from src.libs.std_python_libs import *
from src.libs.data_science_libs import *

import pprint


def get_metrices(a_row: str) -> dict:
    """Get metrices
    Returns:
    --------
    `a_dict` - dict.\n
    """
    # print(a_row)
    metrices = a_row.split("==>")[1].strip()

    metrices = metrices.split(":")
    metrices = [item.split(" ") for item in metrices]

    metrices = list(itertools.chain.from_iterable(metrices))

    metrices = list(filter(lambda item: len(item) != 0, metrices))

    metrices = np.array(metrices)
    # pprint.pprint(metrices)
    # sys.exit(0)

    metrices_k = metrices[np.arange(0, len(metrices), 2)]
    metrices_k = list(map(str.lower, metrices_k))

    metrices_v = metrices[np.arange(1, len(metrices), 2)]
    metrices_v = list(map(float, metrices_v))
    a_dict = dict(zip(metrices_k, metrices_v))

    # pprint.pprint(a_dict)
    # sys.exit(0)
    return a_dict


def filter_unecessary_lines(models_df: pd.DataFrame, lines: list) -> (list, list, list):
    """filter unecessary lines.
    Returns:
    --------
    `lines_wanted` - list.\n
    `lines_files` - list.\n
    `lines_files_bool` - list.\n
    """

    date_train_list = list(models_df["date_train"].values)

    get_files_names = lambda item: os.path.splitext(
        os.path.basename(item.split(":")[0])
    )[0]
    lines_files = list(map(get_files_names, lines))

    get_files_names = lambda item: item in date_train_list
    lines_files_bool = list(map(get_files_names, lines_files))

    get_wanted_data = lambda item: lines_files[item[0]]
    lines_wanted = list(filter(get_wanted_data, enumerate(lines)))

    lines_wanted = list(map(lambda item: item[1], lines_wanted))
    return lines_wanted, lines_files, lines_files_bool


def merge_performace_w_models_data(args, models_df:pd.DataFrame, verbose:int = 0) -> pd.DataFrame:
    """Merge performacne w model data.
    Returns:
    --------
    `a_df` - pd.DataFrame, updated version.\n
    """
    lines = None
    with open(args.performances_path, "r") as f:
        lines = f.read().split("\n")
        pass
    # pprint.pprint(lines)
    
    lines_wanted, lines_files, lines_files_bool =  \
        filter_unecessary_lines(models_df, lines)
    
    a_df = copy.deepcopy(models_df)
    for lw, lf, lfb in zip(lines_wanted, lines_files, lines_files_bool):
        if lfb is False: continue
        a_dict = get_metrices(a_row=lw)
        res = a_df[a_df["date_train"] == lf]
        res["psnr"] = float(a_dict["psnr"])
        res["ssim"] = float(a_dict["ssim"])
        res["time"] = float(a_dict["time"])
        res["mse"] = float(a_dict["mse"])
        index = res.index
        a_df.iloc[index,:] = res
        pass
    target_cols = ["psnr", "ssim", "mse", "time", "size_byte_th", "nbits"]
    for k in target_cols:
        if k not in a_df.columns: continue
        try:
            values = models_df[k].values
            a_df[k] = list(map(float, values))
        except:
            pass
        pass
    if verbose > 0 :
        print(a_df[target_cols].head(5))
    return a_df
