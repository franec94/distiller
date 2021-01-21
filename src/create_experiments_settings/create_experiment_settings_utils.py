# ---------------------------------------------- #
# Python's Std, Community Packages
# ---------------------------------------------- #
from src.libraries.all_libs import *

UNDESIRED_COLUMNS: list = "Unnamed: 0,unnamed: 0,Unnamed 0".split(",")

# =================================================================================================== #
# Create experiments
# =================================================================================================== #

def show_train_cmd():
    cmd="""
CUDA_VISIBLE_DEVICES=0
python3 siren_main_app.py
    --logging_root {LOGGING_ROOT}
    --experiment_name 'quant_train'
    --sidelength {SIDELENGTH}
    --n_hf {N_HF} 
    --n_hl {N_HL} 
    --seed 0
    --cuda
    --evaluate
    --num_epochs {EPOCHS}
    --lr {LR}
    --epochs_til_ckpt 5000
    --resume-from {INITIALIZED_MODEL} 
    --compress {COMPRESS_SCHEDULE}
    --save_test_data_to_csv_path {RESULTS_CSV_PATH}
    --verbose 0
"""

    cmd_2="""echo '
CUDA_VISIBLE_DEVICES=0 \
python3 siren_main_app.py \
    --logging_root {LOGGING_ROOT} \
    --experiment_name 'quant_train' \
    --sidelength {SIDELENGTH} \
    --n_hf {N_HF}  \
    --n_hl {N_HL} \
    --seed 0 \
    --cuda \
    --evaluate \
    --num_epochs {EPOCHS} \
    --lr {LR} \
    --epochs_til_ckpt 5000 \
    --resume-from {INITIALIZED_MODEL} \
    --compress {COMPRESS_SCHEDULE} \
    --save_test_data_to_csv_path {RESULTS_CSV_PATH} \
    --verbose 0'
"""
    print(cmd)
    
    print(re.sub(r"\s+", " ", cmd_2))
    # print(cmd_2)
    pass


def get_workload_infos(conf_dict:dict, bp_conf_dict: dict) -> None:
    """TODO COMMENT IT."""
    hyper_params_grid_list: list = \
        create_hyper_params_combinations(conf_dict["exp_hyper_params_space"])
    hyper_params_train = conf_dict["exp_train_confs"]
    for k in hyper_params_train.keys():
        hyper_params_train[k] = eval(hyper_params_train[k])
        pass
    hyper_params_train = list(ParameterGrid(hyper_params_train))

    hpt_arr = np.array(hyper_params_train)
    oc_arr = np.array(hyper_params_grid_list)
    pos_hp, pos_oc = hpt_arr.shape[0], oc_arr.shape[0]
    total = pos_hp * pos_oc

    data_tb = dict(
        total_hyper_params_combs=oc_arr.shape[0],
        total_train_combs=hpt_arr.shape[0],
        total=oc_arr.shape[0] * hpt_arr.shape[0],
        blueprint_scheduler=conf_dict["dataset"]["blueprint_conf_filename"],
        category_experiment=conf_dict["dataset"]["category_experiment"],
    )
    meta_tb = dict(
        tabular_data=data_tb.items(),
    )
    a_tb = tabulate.tabulate(**meta_tb)
    print(a_tb)

    show_train_cmd()
    pass


def create_hyper_params_combinations(hyper_params_conf_dict: dict) -> list:
    """TODO COMMENT IT."""
    hyper_params_conf_dict_tmp = dict()
    for k, v in hyper_params_conf_dict.items():
        # pprint(eval(v["choices"]))
        choices = eval(v["choices"])
        if "overrides" in k:
            exclude = [eval(v["exclude"])]
            choices = list(zip(choices, exclude * len(choices)))
            pprint(choices)
            pass
        hyper_params_conf_dict_tmp[k] = choices
    hyper_params_grid = list(ParameterGrid(hyper_params_conf_dict_tmp))
    # pprint(hyper_params_grid[0:5])
    # sys.exit(0)
    return hyper_params_grid


def get_target(key_path: list, a_conf_dict: dict):
    """TODO COMMENT IT."""
    tmp_var = key_path[0]
    for a_key in key_path[1:]:
        tmp_var = tmp_var[a_key]
    tmp_var
    pass


def update_target(key_path: list, a_conf_dict: dict, target_val) -> None:
    """TODO COMMENT IT."""
    tmp_var = a_conf_dict[key_path[0]]
    for a_key in key_path[1:len(key_path)-1]:
        if a_key == "all":
            for pos in tmp_var.keys():
                if int(pos.split(".")[1]) not in target_val[1]:
                    tmp_var[pos][key_path[-1]] = target_val[0]
                else:
                    tmp_var[pos][key_path[-1]] = None
            return
        else:
            tmp_var = tmp_var[a_key]
            pass
        pass
    # pprint(tmp_var)
    tmp_var[key_path[-1]] = target_val
    # tmp_var = target_val
    pass


def update_bp_confs(conf_dict:dict, bp_conf_dict: dict) -> list:
    """TODO COMMENT IT."""
    out_conf_list = []

    def conver_key_chuncks(chuncks_list:list):
        out_chuncks_list: list = []
        for a_chunck in chuncks_list:
            # print(a_chunck)
            try:
                a_chunck = int(a_chunck)
            except:
                pass
            out_chuncks_list.append(a_chunck)
        return out_chuncks_list
    
    hyper_params_grid_list: list = \
        create_hyper_params_combinations(conf_dict["exp_hyper_params_space"])

    with tqdm.tqdm(total=len(hyper_params_grid_list)) as pbar:
        pbar.write("Create files...")
        for a_hp_conf in hyper_params_grid_list:
            out_bp_conf_dict = copy.deepcopy(bp_conf_dict)
            for k, v in a_hp_conf.items():
                key_path:list = k.split(".") # ; print(key_path)
                key_path = conver_key_chuncks(key_path) # ; pprint(["{} - {}".format(a_key, type(a_key)) for a_key in key_path])
                update_target(key_path=key_path,
                    a_conf_dict=out_bp_conf_dict, target_val=v)
                pass
            # pprint(out_bp_conf_dict)
            out_conf_list.append(out_bp_conf_dict)
            pbar.update(1)
            pass
    return out_conf_list


def save_created_conf_files(conf_dict: dict, out_conf_list: list, ts = None) -> None:
    """TODO COMMENT IT."""
    out_dir = conf_dict["out_dir"]
    out_dir_path = os.path.join(
            out_dir["root_dir"],
            out_dir["main_out_dir_exp_set"],
            out_dir["category_experiment"])
    if ts:
        out_dir_path = os.path.join(out_dir_path, f"exp_{ts}", "schedulers")
    if not os.path.exists(out_dir_path) or not os.path.isdir(out_dir_path):
        try: os.makedirs(out_dir_path)
        except: pass
        pass

    category_experiment = out_dir["category_experiment"]
    with tqdm.tqdm(total=len(out_conf_list)) as pbar:
        pbar.write("Write output results...")
        for ii, out_conf_instance in enumerate(out_conf_list):
            file_name = f"file_{category_experiment}_{ii}.yaml"
            out_file_path = os.path.join(out_dir_path, file_name)
            with open(f"{out_file_path}", 'w') as out_file:
                _ = yaml.dump(out_conf_instance, out_file) # documents
                pass
            pbar.update(1)
            pass
        pass
    return out_dir_path

# =================================================================================================== #
# Run experiments
# =================================================================================================== #

def get_custom_command(a_conf_train_dict:dict, out_conf:dict, echo: bool = False) -> str:
    """TODO COMMENT IT."""
    LOGGING_ROOT = a_conf_train_dict["logging_root"]
    SIDELENGTH = a_conf_train_dict["sidelength"]
    N_HF = a_conf_train_dict["n_hf"]
    N_HL = a_conf_train_dict["n_hl"]

    LR = a_conf_train_dict["lr"]

    delta_end_epochs = a_conf_train_dict["delta_end_epochs"]
    ending_epoch = out_conf["policies"][0]["ending_epoch"]
    EPOCHS = ending_epoch + delta_end_epochs

    INITIALIZED_MODEL = a_conf_train_dict["init_model"]
    COMPRESS_SCHEDULE = "../schedulers/schedule.yaml"
    RESULTS_CSV_PATH = a_conf_train_dict["results_csv_path"]

    if "lambda_L_1" not in a_conf_train_dict.keys():
        a_conf_train_dict["lambda_L_1"] = 0.0
        pass
    if "lambda_L_2" not in a_conf_train_dict.keys():
        a_conf_train_dict["lambda_L_2"] = 0.0
        pass
    lambda_L_1 = a_conf_train_dict["lambda_L_1"]
    lambda_L_2 = a_conf_train_dict["lambda_L_2"]

    if echo:
        cmd=f"""echo '
CUDA_VISIBLE_DEVICES=0 \
python3 siren_main_app.py \
    --logging_root {LOGGING_ROOT} \
    --experiment_name 'quant_train' \
    --sidelength {SIDELENGTH} \
    --n_hf {N_HF}  \
    --n_hl {N_HL} \
    --seed 0 \
    --cuda \
    --evaluate \
    --num_epochs {EPOCHS} \
    --lr {LR} \
    --lambda_L_1 {lambda_L_1} \
    --lambda_L_2 {lambda_L_2} \
    --epochs_til_ckpt 5000 \
    --resume-from {INITIALIZED_MODEL} \
    --compress {COMPRESS_SCHEDULE} \
    --save_test_data_to_csv_path {RESULTS_CSV_PATH} \
    --verbose 0'
"""
    else:
        cmd=f"""
CUDA_VISIBLE_DEVICES=0 \
python3 siren_main_app.py \
    --logging_root {LOGGING_ROOT} \
    --experiment_name 'quant_train' \
    --sidelength {SIDELENGTH} \
    --n_hf {N_HF}  \
    --n_hl {N_HL} \
    --seed 0 \
    --cuda \
    --evaluate \
    --num_epochs {EPOCHS} \
    --lr {LR} \
    --lambda_L_1 {lambda_L_1} \
    --lambda_L_2 {lambda_L_2} \
    --epochs_til_ckpt 5000 \
    --resume-from {INITIALIZED_MODEL} \
    --compress {COMPRESS_SCHEDULE} \
    --save_test_data_to_csv_path {RESULTS_CSV_PATH} \
    --verbose 0
"""
        pass
    return cmd


def get_dataset_record(tmp_record: dict, exp_train_confs: dict) -> dict:

    src_keys = "n_hf,n_hl,sidelength,sidelength,num_epochs,lr,lambda_L_1,lambda_L_2".split(",")
    dst_keys = "n_hf,n_hl,w,h,num_epochs,lr,L1,L2".split(",")

    assert len(src_keys) == len(dst_keys), f"Error: len(src_keys) != len(dst_keys), that is, {len(src_keys)} != {len(dst_keys)}"

    a_record = copy.deepcopy(tmp_record)

    if "lambda_L_1" not in exp_train_confs.keys():
        exp_train_confs["lambda_L_1"] = 0.0
        pass
    if "lambda_L_2" not in exp_train_confs.keys():
        exp_train_confs["lambda_L_2"] = 0.0
        pass
    if "num_epochs" not in exp_train_confs.keys():
        exp_train_confs["num_epochs"] = 0.0
        pass

    for src_k, dst_k in zip(src_keys, dst_keys):
        a_record[dst_k] = exp_train_confs[src_k]
        pass
    return a_record


def run_subprocess_waiting_it(a_conf_train_dict:dict, file_conf:str, echo: bool = False, verbose: int = 0) -> None:
    """TODO COMMENT IT."""
    cmd = get_custom_command(a_conf_train_dict, file_conf, echo=echo)
    # print(re.sub(r"\s+", " ", cmd))
    process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode
    if verbose == 1:
        print(out.decode())
        print(err.decode())
        print(errcode)
        pass
    pass


def run_subprocess_cmd_waiting_it(cmd:str, verbose: int = 0) -> None:
    """TODO COMMENT IT."""
    # print(re.sub(r"\s+", " ", cmd))
    process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode
    if verbose == 1:
        print(out.decode())
        print(err.decode())
        print(errcode)
        pass
    pass


def write_to_csv(file_name: str, a_df: pd.DataFrame) -> None:
    """TODO COMMENT IT."""
    dir_name = os.path.dirname(file_name)
    try:
        os.makedirs(dir_name)
    except: pass
    a_df.to_csv(file_name, index=False)
    pass


def create_dataset_experiments(args, conf_dict: dict, out_conf_list: list, echo:bool=False, verbose:int=0) -> (pd.DataFrame, dict):
    """TODO COMMENT IT."""

    _debug_mode = True

    hyper_params_train = conf_dict["exp_train_confs"]
    for k in hyper_params_train.keys():
        hyper_params_train[k] = eval(hyper_params_train[k])
        pass
    hyper_params_train = list(ParameterGrid(hyper_params_train))
    # pprint(hyper_params_train[0:5])

    dataset_columns = []
    dataset_columns += conf_dict["dataset"]["dataset_columns_time"].split(",")
    dataset_columns += conf_dict["dataset"]["dataset_columns_scores"].split(",")
    dataset_columns += conf_dict["dataset"]["dataset_columns_settings"].split(",")

    tmp_record = dict(zip(dataset_columns, ["-"] * len(dataset_columns)))

    hpt_arr = np.array(hyper_params_train)
    oc_arr = np.array(out_conf_list)
    pos_hp, pos_oc = hpt_arr.shape[0], oc_arr.shape[0]
    total = pos_hp * pos_oc
    
    ts = time.time()
    if conf_dict["dataset"]["save_dataset"]:
        out_dir_path = save_created_conf_files(conf_dict, out_conf_list, ts)
        pass

    records_list: list = []
    with tqdm.tqdm(total=total) as pbar:
        pbar.write("Creating dataset...")
        
        for hp_train in hpt_arr[:pos_hp]:
            for out_conf in oc_arr[:pos_oc]:
                a_record = get_dataset_record(tmp_record, hp_train)
                cmd = get_custom_command(hp_train, out_conf, echo=True)
                cmd_ = re.sub(r"\s+", " ", cmd)
                a_record["experiment_date"] = f"{ts}"
                a_record["command_line"] = cmd_
                a_record["scheduler"] = out_conf
                a_record["model_name"] = os.path.basename(hp_train["init_model"])

                a_record["model_name"] = os.path.basename(hp_train["init_model"])

                pprint(hp_train)

                delta_end_epochs = hp_train["delta_end_epochs"]

                ending_epoch = out_conf["policies"][0]["ending_epoch"]
                a_record["num_epochs"] = ending_epoch + delta_end_epochs

                pprint(a_record)
                # sys.exit(0)


                # run_subprocess_waiting_it(hp_train, out_conf, echo=True, verbose=1)
                
                records_list.append(a_record)
                pbar.update(1)
                pass
            pass
        pass
    a_df = pd.DataFrame(records_list)

    if _debug_mode:
        dates_cols = conf_dict["dataset"]["dataset_columns_time"].split(",")
        print(a_df[dates_cols].head(2))

        scores_cols = conf_dict["dataset"]["dataset_columns_scores"].split(",")
        print(a_df[scores_cols].head(2))

        train_settings = conf_dict["dataset"]["dataset_columns_settings"].split(",")
        print(a_df[train_settings].head(2))
        pass

    save_dataset: bool = conf_dict["dataset"]["save_dataset"]
    out_dict_info = None
    if save_dataset:
        out_dir = conf_dict["out_dir"]
        root_dir = os.path.join(
            out_dir["root_dir"],
            out_dir["main_out_dir_exp_set"],
            out_dir["category_experiment"], f"exp_{ts}")
        configs_dir = os.path.join(root_dir, "configs")
        fname = os.path.basename(args.conf_file)
        try: os.makedirs(configs_dir)
        except: pass
        shutil.copy(args.conf_file, os.path.join(configs_dir, fname))
        file_name = os.path.join(root_dir, "out", f"out_{ts}.csv")
        write_to_csv(file_name, a_df)

        out_dict_info = dict(
            root_dir=root_dir,
            schedulers=out_dir_path,
            configs_dir=configs_dir,
            configs_file=os.path.join(configs_dir, fname),
            dataset_file_name=file_name,
            total_trials=a_df.shape[0],
            total_schedulers=oc_arr.shape[0]
        )
    else:
        out_dict_info_tmp = dict(
            total_trials=a_df.shape[0],
            total_schedulers=oc_arr.shape[0]
        )
        meta_tb = dict(
            tabular_data=out_dict_info_tmp.items()
        )
        table = tabulate.tabulate(**meta_tb)
        print(table)
        pass


    return a_df, out_dict_info


def run_dataset_experiments(args, conf_dict: dict, a_df:pd.DataFrame = pd.DataFrame(), out_dict_info:dict = None, echo:bool=False, verbose:int=0) -> dict:
    """TODO COMMENT IT."""

    _debug_mode = False

    out_dir = conf_dict["out_dir"]
    out_dir_path = os.path.join(
            out_dir["root_dir"],
            out_dir["main_out_dir_exp_set"],
            out_dir["category_experiment"])
    if out_dict_info is None:
        ts = time.time()
        out_dict_info = dict()
        out_dir_path = os.path.join(out_dir_path, f"exp_{ts}")
        out_dict_info["root_dir"] = out_dir_path
        pass
    if not os.path.exists(out_dir_path) or not os.path.isdir(out_dir_path):
        try: os.makedirs(out_dir_path)
        except: pass
        pass


    if conf_dict["actions"]["dataset_path"]:
        data_path = conf_dict["actions"]["dataset_path"]
        a_df = pd.read_csv(data_path)
        for u_col in UNDESIRED_COLUMNS:
            if u_col in a_df.columns:
                a_df = a_df.drop([u_col], axis=1)
                pass
            pass
        pass
    if a_df.shape == (0, 0): return

    if _debug_mode:
        dataset_columns = []
        dataset_columns += conf_dict["dataset"]["dataset_columns_time"].split(",")
        dataset_columns += conf_dict["dataset"]["dataset_columns_scores"].split(",")
        dataset_columns += conf_dict["dataset"]["dataset_columns_settings"].split(",")

        dates_cols = conf_dict["dataset"]["dataset_columns_time"].split(",")
        print(a_df[dates_cols].head(2))

        scores_cols = conf_dict["dataset"]["dataset_columns_scores"].split(",")
        print(a_df[scores_cols].head(2))

        train_settings = conf_dict["dataset"]["dataset_columns_settings"].split(",")
        print(a_df[train_settings].head(2))
        sys.exit(0)
        pass

    with tqdm.tqdm(total=a_df.shape[0]) as pbar:
        pbar.write("Running dataset exp...")
        keys = list(a_df.columns) # ; pprint(keys)
        for a_row in a_df.values:
            # pprint(a_row)
            # print(type(a_row_dict))
            # print(a_row_dict["command_line"])
            a_row_dict = dict(zip(keys, a_row))

            scheduler = a_row_dict["scheduler"]
            scheduler_file_name = os.path.join("../schedulers/scheduler.yaml")
            try: os.makedirs("../schedulers/")
            except: pass
            
            with open(f"{scheduler_file_name}", 'w') as scheduler_file:
                _ = yaml.dump(scheduler, scheduler_file)
                pass
            cmd = a_row_dict["command_line"]
            print(cmd)
            run_subprocess_cmd_waiting_it(cmd, verbose=1)
            pbar.update(1)
            sys.exit(0)
            pass
        pass
    if 'configs_dir' not in out_dict_info.keys():
        root_dir = out_dir_path
        configs_dir = os.path.join(root_dir, "configs")
        out_dict_info["configs_dir"] = configs_dir
        fname = os.path.basename(args.conf_file)
        try: os.makedirs(configs_dir)
        except: pass
        shutil.copy(args.conf_file, os.path.join(configs_dir, fname))
        pass
    if 'total_trials' not in out_dict_info.keys():
        out_dict_info["total_trials"] = a_df.shape[0]
        pass
    if type(conf_dict["exp_train_confs"]["results_csv_path"]) is str:
        conf_dict["exp_train_confs"]["results_csv_path"] = eval(conf_dict["exp_train_confs"]["results_csv_path"])
    out_dict_info["results_csv_path"] = conf_dict["exp_train_confs"]["results_csv_path"][0]
    return out_dict_info