from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.data_loaders import dataset_loaders

def get_model_data(n_hf=64, n_hl=5) -> (np.array, np.array):
    """Get model characteristics from its number of hidden features, and number of hidden layers established.\n
    Args:
    -----
    `n_hf` - int, number of hidden features.\n
    `n_hl` - int, number of hidden layers.\n
    Returns:
    --------
    `wgts_arr` - np.array, list of hidden weights plus input layer and output layers weights.\n
    `biases_arr` - np.array, list of hidden biases plus input layer and output layer biases.\n
    """
    wgts_input = [n_hf * 2]
    wgts_hidden = [n_hf * n_hf] * n_hl
    wgts_output = [n_hf]
    wgts_list = wgts_input + wgts_hidden + wgts_output
    wgts_arr = np.array(wgts_list)
    
    b_input = [2]
    b_hidden = [n_hf] * n_hl
    b_output = [1]
    b_list = b_input + b_hidden + b_output
    biases_arr = np.array(b_list)
    return wgts_arr, biases_arr


def get_sample_from_blueprint() -> collections.OrderedDict: 
    # def get_sample_from_blueprint() -> dict: 
    """Get ordered dict example
    Returns:
    --------
    `a_sample` - collections.OrderedDict.\n
    """
    a_sample = collections.OrderedDict(
        # a_sample = dict(
        experiment_date=None,
        date_train=None,
        date_test=None,
        init_from=None,
        root_dir=None,
        model_name=None,
        size_byte=None,
        footprint=None,
        psnr=None,
        bpp=None,
        CR=None,
        mse=None,
        ssim=None,
        time=None,
        entropy=None,
        scheduler_name=None,
        scheduler=None,
        prune_techs=None,
        prune_rate=None,
        quant_techs=None,
        command_line=None,
        num_epochs=None,
        n_hf=None,
        n_hl=None,
        w=None,
        h=None,
        L1=None,
        L2=None,
        lr=None,
        size_byte_th=None,
        experiment_date_2=None,
        nbits=None,
        image_name=None,
    )

    for k,v in a_sample.items():
        a_sample[k] = '-'
    return a_sample


def process_scheduler_file(args, a_root: str, a_sample: collections.OrderedDict) -> collections.OrderedDict:
    """Process scheduler file.
    Args:
    -----
    `args` - Namespace object.\n
    `a_root` - str python object, root directory.\n
    `a_sample` - collections.OrderedDict.\n
    Returns:
    --------
    `a_sample` - collections.OrderedDict, updated instance.\n
    """

    n_hf: int = a_sample["n_hf"]
    n_hl: int = a_sample["n_hl"]

    prune_rates: np.array = np.zeros(n_hl+2)

    wgts_arr, biases_arr = get_model_data(n_hf=n_hf, n_hl=n_hl)

    configs_dirpath = os.path.join(a_root, "configs")
    for a_root, dirs_list, files_list in os.walk(configs_dirpath):
        scheduler = os.path.join(a_root, files_list[0])
        pass
    
    with open(scheduler, "r") as s_fp:
        # scheduler_dict = yaml.load(s_fp, loader=yaml.FullLoader)
        scheduler_dict = yaml.load(s_fp,)
        pass
    # pprint.pprint(scheduler_dict)
    # sys.exit(0)

    quant_classes: set = set()

    for k, v in scheduler_dict["quantizers"].items():
        quant_techs = v["class"]
        quant_classes.add(quant_techs)
        overrides = v["overrides"]
        for k_layer, v_layer in overrides.items():
            if v_layer["bits_weights"]:
                nbits = v_layer["bits_weights"]
                break
            pass
        pass
    
    tot = np.sum(wgts_arr) + np.sum(biases_arr)
    tot_saved = \
        np.sum(wgts_arr - wgts_arr * prune_rates) + \
        np.sum(biases_arr)
    a_sample["prune_rate"] = -1
    a_sample["size_byte_th"] = -1
    a_sample["nbits"] = nbits
    a_sample["quant_techs"] = '+'.join(list(quant_classes))
    a_sample["scheduler"] = str(scheduler_dict)

    return a_sample


def process_log_file_for_cmd_line(args, a_root, files_list: list) -> dict:
    """Process log file.
    Args:
    -----
    `args` - Namespace object.\n
    `a_root` - str python object, root directory.\n
    `files_list` - list of file from whci to sample.\n
    Returns:
    --------
    Returns:
    --------
    `cmd_line_data_dict` - python3 dict object. Default None if something went wrong.\n
    """

    # Gte log file.
    get_logfile = lambda a_f: os.path.splitext(a_f)[1] == ".log"
    logs = list(filter(get_logfile, files_list))
    if len(logs) == 0: return None
    if len(logs) != 1: return None

    a_log = logs[0]
    a_log_path = os.path.join(a_root, a_log)
    # print(a_log)

    with open(a_log_path, "r") as fp:
        lines = fp.read().split("\n")
        target_item = "Command line:"
        def get_cmd_line(item, target_item=target_item): return target_item in item
        cmd_line = list(filter(get_cmd_line, lines))
        if len(cmd_line) == 0: return None
        if len(cmd_line) != 1: return None
        a_cmd_line = cmd_line[0]
        try:
            a_cmd_line = a_cmd_line.split(f"{target_item}")[1].strip()
            # pprint.pprint(a_cmd_line)
        except:
            return None
        pass
    
       
    a_cmd_line_opts = a_cmd_line.split("--")

    num_epochs = list(filter(lambda item: item.startswith("num_epochs"), a_cmd_line_opts))[0]
    num_epochs = int(num_epochs.split(" ")[1])

    n_hf = list(filter(lambda item: item.startswith("n_hf"), a_cmd_line_opts))[0]
    n_hf = int(n_hf.split(" ")[1])

    n_hl = list(filter(lambda item: item.startswith("n_hl"), a_cmd_line_opts))[0]
    n_hl = int(n_hl.split(" ")[1])

    lr = list(filter(lambda item: item.startswith("lr"), a_cmd_line_opts))[0]
    lr = float(lr.split(" ")[1])

    L1 = list(filter(lambda item: item.startswith("lambda_L_1"), a_cmd_line_opts))[0]
    L1 = float(L1.split(" ")[1])

    L2 = list(filter(lambda item: item.startswith("lambda_L_2"), a_cmd_line_opts))[0]
    L2 = float(L2.split(" ")[1])

    sidelength = list(filter(lambda item: item.startswith("sidelength"), a_cmd_line_opts))[0]
    sidelength = int(sidelength.split(" ")[1])

    w = sidelength
    h = sidelength

    get_model_name = lambda a_f: os.path.splitext(a_f)[1] == ".tar"
    model_names = list(filter(get_model_name, files_list))
    
    get_model_name = lambda a_f: a_f.endswith("_checkpoint.pth.tar")
    get_model_name = lambda a_f: a_f.endswith("_best.pth.tar")
    try:
        model_name = list(filter(get_model_name, files_list))[0]
    except: pass
    get_model_name = lambda a_f: os.path.basename(a_f).startswith("final_epoch_final_ckpt_epoch_")
    try:
        model_name = list(filter(get_model_name, files_list))[0]
    except: pass

    # print(a_root)
    # print(model_name)
    # sys.exit(0)

    cmd_line_data_dict: dict = dict(
        command_line = a_cmd_line,
        num_epochs=num_epochs,
        root_dir=a_root,
        model_name=model_name,
        n_hf=n_hf,
        n_hl=n_hl,
        lr=lr,
        L1=L1,
        L2=L2,
        w=w,
        h=h,
        date_train=os.path.splitext(os.path.basename(a_log))[0]
    )

    return cmd_line_data_dict


def get_directories_of_trained_models(args) -> list:
    """Get directories of trained models.
    Args:
    -----
    `args` - Namespace object.\n
    Returns:
    --------
    `yelded list` - yelded list.\n
    """

    data: list = []
    for a_root, dirs_list, files_list in os.walk(args.root_dir):
        if a_root.endswith("configs"): continue
        # print(a_root)

        a_train_data = [a_root, dirs_list, files_list]
        yield a_train_data
    pass


def save_dataset_as_csv(args, all_samples: list, out_filename="out.csv", verbose=0) -> pd.DataFrame:
    """Save dataset as csv.
    Args:
    -----
    `args` -  arguments for leading data storing process.\n
    `all_samples` - list object.\n
    Returns:
    --------
    `a_df` - pd.DataFrame created.\n
    """
    if len(all_samples) != 0:
        a_sample = get_sample_from_blueprint()
        columns = list(a_sample.keys())

        a_df_path = os.path.join(args.out_dir, f"{out_filename}")
        a_df = pd.DataFrame(data=all_samples, columns=columns)
        if verbose > 0:
            print(a_df.head(5))
        a_df.to_csv(a_df_path)
        pass
    return a_df


def get_data_as_list(args) -> list:
    """Get data for constructing out dataset as list.
    Args:
    -----
    `args` -  arguments for leading data retrieving process.\n
    Returns:
    --------
    `all_samples` - list object.\n
    """
    all_samples: list = []
    for a_root, dirs_list, files_list in get_directories_of_trained_models(args=args):

        cmd_line_data_dict = \
            process_log_file_for_cmd_line(args=args, a_root=a_root, files_list=files_list)
        if not cmd_line_data_dict: continue

        a_sample = get_sample_from_blueprint()
        for k,v in cmd_line_data_dict.items():
            if k not in a_sample.keys(): continue
            a_sample[k] = v
            pass
        process_scheduler_file(args=args, a_root=a_root, a_sample=a_sample) 

        all_samples.append(a_sample)
        pass
    return all_samples


def get_pruned_params_numbers(a_row, n_hf=64, n_hl = 5) -> (np.array, np.array):
    scheduler: str = a_row["scheduler"].values[0]
    root_dir = a_row["root_dir"].values[0]
    if scheduler == "-":
        scheduler_path = os.path.join(root_dir, "configs", "siren64_5.schedule_agp.yaml")
        with open(scheduler_path, "r") as f:
            scheduler_dict = yaml.load(f)
            pass
        pass
    else:
        scheduler_dict = eval(scheduler)
        pass

    prune_rates = np.zeros(n_hl+2)
    for kp, pv in scheduler_dict["pruners"].items():
        final_sparsity = pv["final_sparsity"]
        for wgt_name in  pv["weights"]:
            wgt_pos = int(wgt_name.split(".")[2])
            prune_rates[wgt_pos] = final_sparsity
            pass
        pass

    wgts_arr, biases_arr = \
        get_model_data(n_hf=n_hf, n_hl=n_hl)
    return (wgts_arr - wgts_arr * prune_rates), biases_arr


def get_which_quanted_params(scheduler, n_hf=64, n_hl = 5) -> (np.array, np.array):
    
    wgts_quanted_rates = np.ones(n_hl+2) * 32
    biases_quanted_rates = np.ones(n_hl+2) * 32

    # pprint(scheduler)
    
    if not scheduler or scheduler == "-" or scheduler == "":
        return wgts_quanted_rates, biases_quanted_rates

    scheduler_dict = eval(scheduler)
    

    
    linear_quantizer = scheduler_dict["quantizers"]["linear_quantizer"]
    for kp, pv in linear_quantizer["overrides"].items():
        bits_weights = pv["bits_weights"]
        bits_bias = pv["bits_bias"]
        wgt_pos = int(kp.split(".")[1])
        if bits_weights:
            wgts_quanted_rates[wgt_pos] = bits_weights
        if bits_bias:
            biases_quanted_rates[wgt_pos] = bits_bias
        pass
    return wgts_quanted_rates, biases_quanted_rates


def adjust_out_dataframe_by_conf_data(conf_data: dict, a_df: pd.DataFrame, n_hf=64, n_hl=5) -> pd.DataFrame:

    adjusted_df = copy.deepcopy(a_df)
    if adjusted_df.shape[0] == 0: return adjusted_df

    date_str:str = conf_data["init_from"]["date"]
    image_name:str = conf_data["input_data"]["image_name"]

    pruned_image_df = \
        dataset_loaders.load_prunining_dataset(dtype="dataframe", image_name=image_name)

    date_attr: str = "date_train"
    if date_attr not in pruned_image_df.columns:
        date_attr = "date"
    
    if pruned_image_df.shape[0] == 0: return adjusted_df
    if not date_str in pruned_image_df[f"{date_attr}"].unique(): return adjusted_df

    # pick_cols = "size_byte_th,prune_rate,prune_techs".split(",")
    pos = pruned_image_df[f"{date_attr}"] == date_str
    a_row = pruned_image_df[pos].head(1)

    adjusted_df["image_name"] = [image_name] * a_df.shape[0]
    adjusted_df["init_from"] = [date_str] * a_df.shape[0]

    
    def calculate_size_byte_th(scheduler, a_row=a_row, n_hf=n_hf, n_hl=n_hl):
        try:
            wgts_arr_pruned, biasess_arr_pruned = \
                get_pruned_params_numbers(a_row, n_hf=n_hf, n_hl = n_hl)
            
            wgts_quanted_rates, biases_quanted_rates = \
                get_which_quanted_params(scheduler, n_hf=n_hf, n_hl = n_hl)
            
            wgts = wgts_arr_pruned * wgts_quanted_rates
            biases = biasess_arr_pruned * biases_quanted_rates

            params = np.concatenate([wgts, biases])
            no_params = np.sum(params)
            # print(no_params)
            return no_params / 8
        except:
            size_byte = a_row["size(byte)"].values[0]
            ilayer_byte = n_hf * 2 * 32 + 2 * 32
            olayer_byte = n_hf * 32 + 1 * 32
            hbiases_byte = n_hf * n_hl * 32
            size_bit_hidden = size_byte * 8 - ( ilayer_byte + olayer_byte + hbiases_byte)
            no_params_per_hidden_layer = size_bit_hidden / 32 / n_hl
            
            _, biasess_arr_pruned = get_model_data(n_hf=n_hf, n_hl=n_hl)
            
            wgts_quanted_rates, biases_quanted_rates = \
                get_which_quanted_params(scheduler, n_hf=n_hf, n_hl = n_hl)
            
            wgts_arr_pruned = [n_hf*2] + ([no_params_per_hidden_layer] * n_hl) + [n_hf]
            wgts_arr_pruned = np.array(wgts_arr_pruned)

            wgts = wgts_arr_pruned * wgts_quanted_rates
            biases = biasess_arr_pruned * biases_quanted_rates

            params = np.concatenate([wgts, biases])
            no_params = np.sum(params)
            # print(no_params)
            return no_params / 8
    values = adjusted_df["scheduler"].values
    adjusted_df["size_byte_th"] = list(map(calculate_size_byte_th, values))

    prune_techs_attr: str = "prune_techs"
    prune_rate_attr: str = "prune_rate"
    if prune_techs_attr not in a_row.columns:
        pt, pr = a_row["cmprss-class"].values[0].split(":")
        a_row[f"{prune_techs_attr}"] = [pt] * a_row.shape[0]
        a_row[f"{prune_rate_attr}"] = [float(pr)] * a_row.shape[0]
        pass

    adjusted_df["prune_techs"] = [a_row["prune_techs"].values[0]] * adjusted_df.shape[0]
    adjusted_df["prune_rate"] = [a_row["prune_rate"].values[0]] * adjusted_df.shape[0]

    def adjust_bpp(item):
        size_byte_th, w, h = item
        return size_byte_th * 8 / (w * h)
    values = adjusted_df[["size_byte_th", "w", "h"]].values
    adjusted_df["bpp"] = list(map(adjust_bpp, values))
    
    return adjusted_df


def create_out_dataset(args, conf_data: dict = None, out_filename="out.csv", n_hf=64, n_hl=5, verbose = 0) -> None:
    """Create out dataset.
    -----
    `args` -  arguments for leading data storing process.\n
    """
    a_df = pd.DataFrame()

    # Get information from all trained models.
    all_samples:list = get_data_as_list(args=args)
    if conf_data:
        if len(all_samples) != 0:
            a_sample = get_sample_from_blueprint()
            columns = list(a_sample.keys())
            a_df = pd.DataFrame(data=all_samples, columns=columns)
            if verbose > 0:
                print(a_df.head(5))
            
            adjusted_df = \
                adjust_out_dataframe_by_conf_data(conf_data=conf_data, a_df=a_df, n_hf=n_hf, n_hl=n_hl)
            if verbose > 0:
                print(adjusted_df.head(5))

            a_df_path = os.path.join(args.out_dir, f"{out_filename}")
            adjusted_df.to_csv(a_df_path)
            return adjusted_df
        pass

    # Save information if any from trained models.
    a_df = save_dataset_as_csv(args=args, all_samples=all_samples, out_filename=out_filename)
    return a_df
