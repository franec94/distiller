from src_cpd.libs.std_libs import *


def get_metrices(a_row):
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

    # print(a_dict)

    # pprint.pprint(a_dict)
    # sys.exit(0)
    return a_dict


def filter_unecessary_lines(models_df, lines):
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


def merge_performace_w_models_data(args, models_df: pd.DataFrame, verbose:int = 0) -> pd.DataFrame:
    """Merge performance with models data."""

    lines = None
    with open(args.performances_path, "r") as f:
        lines = f.read().split("\n")
        pass
    # pprint.pprint(lines)
    
    lines_wanted, lines_files, lines_files_bool =  \
        filter_unecessary_lines(models_df, lines)
    

    for lw, lf, lfb in zip(lines_wanted, lines_files, lines_files_bool):
        if lfb is False: continue
        a_dict = get_metrices(a_row=lw)
        res = models_df[models_df["date_train"] == lf]
        res["psnr"] = a_dict["psnr"]
        res["ssim"] = a_dict["ssim"]
        res["time"] = a_dict["time"]
        res["mse"] = a_dict["mse"]
        index = res.index
        models_df.iloc[index,:] = res
        pass

    pick_cols = ["psnr", "ssim", "mse", "time"]
    for k in pick_cols:
        if k in models_df.columns:
            vals = models_df[k].values
            models_df[k] = list(map(float, vals))
        pass
    if verbose > 0:
        print(models_df[pick_cols].head(5))
    return models_df