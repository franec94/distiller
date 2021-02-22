from src_cpd.libs.std_libs import *


def get_model_data(n_hf=64, n_hl=5):
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
    )

    for k,v in a_sample.items():
        a_sample[k] = '-'
    return a_sample


def process_scheduler_file(args, a_root: str, a_sample: collections.OrderedDict) -> object:
    """Process scheduler file."""

    n_hf = a_sample["n_hf"]
    n_hl = a_sample["n_hl"]

    prune_rates = np.zeros(n_hl+2)

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

    for k, v in scheduler_dict["pruners"].items():
        final_sparsity = v["final_sparsity"]
        weights = v["weights"]
        weights = list(map(lambda item: int(item.split(".")[2]), weights))
        prune_rates[weights] = final_sparsity
        pass
    
    tot = np.sum(wgts_arr) + np.sum(biases_arr)
    tot_saved = \
        np.sum(wgts_arr - wgts_arr * prune_rates) + \
        np.sum(biases_arr)
    a_sample["prune_rate"] = 1 - tot_saved/ tot
    a_sample["size_byte_th"] = tot_saved * 32 / 8
    pass


def process_log_file(args, a_root, files_list: list) -> object:
    """Process log file.
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
    model_name = list(filter(get_model_name, files_list))[0]

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
    """Get directories of trained models."""

    data: list = []
    for a_root, dirs_list, files_list in os.walk(args.root_dir):
        if a_root.endswith("configs"): continue
        # print(a_root)

        a_train_data = [a_root, dirs_list, files_list]
        yield a_train_data
    pass


def save_dataset_as_csv(args, all_samples: list) -> pd.DataFrame:
    """Save dataset as csv.
    Args:
    -----
    `args` -  arguments for leading data storing process.\n
    `all_samples` - list object.\n
    """
    if len(all_samples) != 0:
        a_sample = get_sample_from_blueprint()
        columns = list(a_sample.keys())

        a_df_path = os.path.join(args.out_dir, "out.csv")
        a_df = pd.DataFrame(data=all_samples, columns=columns)
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
            process_log_file(args=args, a_root=a_root, files_list=files_list)
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


def create_out_dataset(args) -> None:
    """Create out dataset.
    -----
    `args` -  arguments for leading data storing process.\n
    """

    # Get information from all trained models.
    all_samples = get_data_as_list(args=args)

    # Save information if any from trained models.
    a_df = save_dataset_as_csv(args=args, all_samples=all_samples)
    return a_df
