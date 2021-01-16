from src.libraries.std_libs import *
from src.libraries.data_science_libs import *
from src.generics.utils import read_conf_file_content
from src.generics.enanching_datasets_utils import *
from src.generics.utils import get_overall_stats_from_input_dirs


def save_dataset(conf_dict: dict, root_trains_list: list, field_names: list) -> (object, pd.DataFrame, str):
    """Save created dataset.
    Args;:
    ------
    `conf_dict` - dict object containing some config options with which decide how to perform dataset storing.\n
    `root_trains_list` - list of object to be inserted within a pd.DataFrame.\n
    `field_names` - list object withi names about dataset attributes/columns.\n
    Returns:
    --------
    `timestamp_file` - timestap related to date-time on which dataset has been created.\n
    `data_trains_df` - created pd.DataFrame.\n
    `dest_file_path_csv` - str object containing file path to location in which dataset has been stored.\n
    """

    timestamp_file, data_trains_df, dest_file_path_csv = None, None, None

    if root_trains_list != []:
        data = list(map(lambda item: item._asdict(), root_trains_list))
        data_trains_df = pd.DataFrame(data = data, columns = field_names)
        print(data_trains_df.head(5))

        print(data_trains_df.info())
        print(data_trains_df["command_line"].head(5))

        timestamp_file = time.time()
        dest_dir_csv = os.path.join(conf_dict["root_dest_results"], f"pool_{timestamp_file}")
        dest_file_path_csv = os.path.join(dest_dir_csv, f"pool_{timestamp_file}.csv")

        if not os.path.isdir(dest_dir_csv):
            try:
                os.makedirs(dest_dir_csv)
                print(f"Dest directory '{dest_dir_csv}' created!")
                pass
            except:
                print(f"Dest directory '{dest_dir_csv}' already exists!")
                pass
        else: 
            print(f"Dest directory '{dest_dir_csv}' already exists!")
            pass
        data_trains_df.to_csv(dest_file_path_csv)
        pass
    return timestamp_file, data_trains_df, dest_file_path_csv


def create_dataset_pruned_models_trains(conf_dict: dict) -> pd.DataFrame:
    """ Create a dataset from already trained models.
    Args:
    -----
    `conf_dict` - dict object with specifications about how to carry out dataset creation.\n
    Return:
    -------
    `pd.DataFrame` - created dataframe.\n
    """
    
    #                       0,        1,        2,       3,         4,        5,        6,   7,  8, 9, 10,  11,            12,       13,         14,        15,         16,          17         18
    field_names = "date_train,date_test,init_from,root_dir,model_name,size_byte,footprint,psnr,bpp,CR,mse,ssim,scheduler_name,scheduler,prune_techs,prune_rate,quant_techs,command_line,num_epochs".split(",")
    data_record_blue_print = ["-","-","-","-","-",0,0,0,0,0,0,0,"-","-","-",0.,"-","-",0]
    assert len(data_record_blue_print) == len(field_names), \
        f"len(data_record) != len(field_names), that is, {len(data_record_blue_print)} != {len(field_names)}"

    TrainRecord = collections.namedtuple("TrainRecord", field_names)
    timestamp_file, data_trains_df, dest_file_path_csv = None, None, None

    _, record_stats_n = get_overall_stats_from_input_dirs(conf_dict, verbose = 1)
    root_trains_list = []
    with tqdm.tqdm(total=record_stats_n.no_dirs) as pbar:
        for a_dir_path in conf_dict['input_dirs_list']:
            for dir_name, subdirs_list, files_list in os.walk(a_dir_path):
                
                if "configs" not in subdirs_list:
                    continue

                data_record = copy.deepcopy(data_record_blue_print)
                

                models_files_list = list(filter(lambda item: item.endswith(".tar"), files_list))
                def get_final_model(item):
                    file_name = os.path.basename(item)
                    return file_name.startswith("_final")
                final_model = list(filter(get_final_model, models_files_list))[0]
                def get_intermediate_model_model(item):
                    file_name = os.path.basename(item)
                    return file_name.startswith("_pruned")
                
                intermediate_models = list(filter(get_intermediate_model_model, models_files_list))
                models_files_list = [final_model] + intermediate_models

                train_logs_list = list(filter(lambda item: item.endswith(".log"), files_list))[0]
                date_train = os.path.basename(dir_name).replace("_", "")

                with open(os.path.join(dir_name, train_logs_list)) as log_fp:
                    lines = log_fp.read().split("\n")
                    command_line = list(filter(lambda item: "Command line" in item, lines))[0]
                    pass

                data_record[17] = command_line # command_line
                data_record[3] = dir_name      # root_dir

                

                configs_subdir = os.path.join(dir_name, "configs")
                scheduler_name = None
                for dir_name, subdirs_list, files_list in os.walk(configs_subdir):
                    scheduler_name = files_list[0]
                    pass
                for ii, a_model in enumerate(models_files_list):
                    data_record[0] = date_train if ii == 0 else f"{date_train}-no_{ii}" # date_train
                    data_record[1] = "-"                                                # date_test
                    data_record[4] = a_model                                            # model_name
                    

                    if scheduler_name and ii == 0:
                        data_record[12] = scheduler_name                           # scheduler_name
                        scheduler_path = os.path.join(configs_subdir, scheduler_name)
                        data_record[13] = read_conf_file_content(scheduler_path)   # scheduler
                    elif a_model.startswith("_pruned"):
                        trail_to_drop = len(".pth.tar")
                        model_name = a_model # os.path.basename(a_model)
                        prune_rate = float(model_name[0:-1-trail_to_drop+1].split("_")[-1])
                        data_record[15] = prune_rate / 100 # prune_rate
                        epochs =  int(model_name[0:-1-trail_to_drop+1].split("_")[4])
                        data_record[18] = epochs     # epochs
                    elif a_model.startswith("_mid_ckpt_epoch_"):
                        # example: _mid_ckpt_epoch_EPOCHS.pth.tar
                        model_name = a_model
                        trail_to_drop = len(".pth.tar")
                        epochs =  int(model_name[0:-1-trail_to_drop+1].split("_")[4])
                        data_record[18] = epochs     # epochs
                        pass

                    a_record = TrainRecord._make(data_record)
                    root_trains_list.append(a_record)
                    pass
                pbar.update(1)
                pass
            pass

        timestamp_file, data_trains_df, dest_file_path_csv = \
            save_dataset(conf_dict = conf_dict, root_trains_list = root_trains_list, field_names = field_names)
        pass

    return dest_file_path_csv, timestamp_file, data_trains_df


def create_dataset_baseline_trains(conf_dict: dict) -> pd.DataFrame:
    """ Create a dataset from already trained models.
    Args:
    -----
    `conf_dict` - dict object with specifications about how to carry out dataset creation.\n
    Return:
    -------
    `pd.DataFrame` - created dataframe.\n
    """
    
    #                       0,        1,        2,       3,         4,        5,        6,   7,  8, 9, 10,  11,            12,       13,         14,        15,         16,          17         18
    field_names = "date_train,date_test,init_from,root_dir,model_name,size_byte,footprint,psnr,bpp,CR,mse,ssim,scheduler_name,scheduler,prune_techs,prune_rate,quant_techs,command_line,num_epochs".split(",")
    data_record_blue_print = ["-","-","-","-","-",0,0,0,0,0,0,0,"-","-","-",0.,"-","-",0]
    assert len(data_record_blue_print) == len(field_names), \
        f"len(data_record) != len(field_names), that is, {len(data_record_blue_print)} != {len(field_names)}"

    TrainRecord = collections.namedtuple("TrainRecord", field_names)
    timestamp_file, data_trains_df, dest_file_path_csv = None, None, None

    _, record_stats_n = get_overall_stats_from_input_dirs(conf_dict, verbose = 1)
    root_trains_list = []
    with tqdm.tqdm(total=record_stats_n.no_dirs) as pbar:
        for a_dir_path in conf_dict['input_dirs_list']:
            for dir_name, subdirs_list, files_list in os.walk(a_dir_path):
                print(f"Processing {dir_name}...")
                for a_file in files_list:
                    print("\t", a_file)
                data_record = copy.deepcopy(data_record_blue_print)
                command_line = '-'

                models_files_list_tmp = list(filter(lambda item: item.endswith(".tar"), files_list))
                models_files_list = []

                try:
                    def get_final_model(item):
                        file_name = os.path.basename(item)
                        return file_name.startswith("_checkpoint")
                    final_model = list(filter(get_final_model, models_files_list_tmp))[0]
                    models_files_list.appen(final_model)
                except:
                    pass
                try:
                    def get_best_model(item):
                        file_name = os.path.basename(item)
                        return file_name.startswith("_best")
                    best_model = list(filter(get_best_model, models_files_list_tmp))[0]
                    models_files_list.appen(best_model)
                except:
                    pass
                try:
                    def get_intermediate_model_model(item):
                        file_name = os.path.basename(item)
                        return file_name.startswith("_mid_ckpt_epoch_")
                    intermediate_models = list(filter(get_intermediate_model_model, models_files_list_tmp))
                    models_files_list.extend(intermediate_models)
                except:
                    pass
                
                try:
                    train_logs_list = list(filter(lambda item: item.endswith(".log"), files_list))[0]
                    with open(os.path.join(dir_name, train_logs_list)) as log_fp:
                        lines = log_fp.read().split("\n")
                        command_line = list(filter(lambda item: "Command line" in item, lines))[0]
                        pass
                except:
                    pass

                data_record[17] = command_line # command_line
                data_record[3] = dir_name      # root_dir
                
                date_train = os.path.basename(dir_name).replace("_", "")
                for ii, a_model in enumerate(models_files_list):
                    data_record[0] = date_train if ii == 0 else f"{date_train}-no_{ii}" # date_train
                    data_record[1] = "-"                                                # date_test
                    data_record[4] = a_model                                            # model_name
                    
                    if a_model.startswith("_mid_ckpt_epoch_"):
                        # example: _mid_ckpt_epoch_EPOCHS.pth.tar
                        model_name = a_model
                        trail_to_drop = len(".pth.tar")
                        epochs =  int(model_name[0:-1-trail_to_drop+1].split("_")[4])
                        data_record[18] = epochs     # epochs
                        pass

                    a_record = TrainRecord._make(data_record)
                    root_trains_list.append(a_record)
                    pass
                pbar.update(1)
                pass
            pass

        timestamp_file, data_trains_df, dest_file_path_csv = \
            save_dataset(conf_dict = conf_dict, root_trains_list = root_trains_list, field_names = field_names)
        pass

    return dest_file_path_csv, timestamp_file, data_trains_df


def add_prune_class(data_trains_cp_df: pd.DataFrame) -> None:
    """Add prune kind technique employed to dataset.
    Args:
    -----
    `data_trains_cp_df` - input dataframe to be updated adding prune kind technique employed.\n
    """
    columns = list(data_trains_cp_df.columns)
    indeces = list(range(len(data_trains_cp_df.columns)))
    key_index_dict = dict(zip(columns, indeces))
    
    calculate_prune_class = \
        get_custom_calculate_prune_class(key_index_dict=key_index_dict)
    data_trains_cp_df["prune_techs"] = \
         list(map(calculate_prune_class, data_trains_cp_df.values))
    pass


def add_prune_rate(data_trains_cp_df: pd.DataFrame) -> None:
    """Add prune rate to prune kind technique employed to dataset.
    Args:
    -----
    `data_trains_cp_df` - input dataframe to be updated adding prune rate fixed for related prune kind technique employed.\n
    """
    columns = list(data_trains_cp_df.columns)
    indeces = list(range(len(data_trains_cp_df.columns)))
    key_index_dict = dict(zip(columns, indeces))
    
    calculate_prune_rate = \
        get_custom_calculate_prune_rate(key_index_dict=key_index_dict)
    data_trains_cp_df["prune_rate"] = \
         list(map(calculate_prune_rate, data_trains_cp_df.values))
    
    pass


def add_cmd_line_options(data_trains_cp_df: pd.DataFrame) -> None:
    """Add models' information about both models' arhcitecture and how train was accomplished.
    Args:
    -----
    `data_trains_cp_df` - input dataframe to be updated.\n
    """
    columns = list(data_trains_cp_df.columns)
    indeces = list(range(len(data_trains_cp_df.columns)))
    key_index_dict = dict(zip(columns, indeces))
    
    get_cmd_line_opts_dict = \
        get_custom_cmd_line_opts_dict(key_index_dict)
    cmd_line_opts_list = list(map(get_cmd_line_opts_dict, data_trains_cp_df.values))
    cmd_line_opts_df = pd.DataFrame(data = cmd_line_opts_list)

    # pprint(cmd_line_opts_df.columns)
    # pprint(data_trains_cp_df.columns)
    # sys.exit(0)

    num_epochs_list = list(cmd_line_opts_df["num_epochs"].values)
    cmd_line_opts_df = cmd_line_opts_df.drop(["num_epochs"], axis = 1)
    data_trains_cp_tmp_df = pd.concat([data_trains_cp_df, cmd_line_opts_df],
        axis = 1, ignore_index = True)
    columns = list(data_trains_cp_df.columns) + list(cmd_line_opts_df.columns)
    data_trains_cp_tmp_df.columns = columns

    if "num_epochs" not in data_trains_cp_df.columns:
        raise Exception(f"{list(data_trains_cp_df.columns)}")

    old_num_epochs_list = list(data_trains_cp_df["num_epochs"].values)
    pair_old_new_epochs = list(zip(old_num_epochs_list, num_epochs_list))
    def map_epochs(item):
        old_e, new_e = item
        if old_e == 0: return new_e
        return old_e
    data_trains_cp_tmp_df["num_epochs"] = list(map(map_epochs, pair_old_new_epochs))
    
    return data_trains_cp_tmp_df


def add_models_size_byte(data_trains_cp_df: pd.DataFrame) -> None:
    """Add models' size for each model trained with input dataset.
    Args:
    -----
    `data_trains_cp_df` - input dataframe to be updated.\n
    """
    columns = list(data_trains_cp_df.columns)
    indeces = list(range(len(data_trains_cp_df.columns)))
    key_index_dict = dict(zip(columns, indeces))

    def get_model_size_byte(a_row, key_index_dict=key_index_dict):
        n_hf, n_hl = int(a_row[key_index_dict["n_hf"]]), int(a_row[key_index_dict["n_hl"]])
        prune_rate = float(a_row[key_index_dict["prune_rate"]])
        
        # Models's derived infos.
        biases_list = np.array([2] + [n_hf] * n_hl + [1])
        wgts_list = np.array([n_hf*2] + [n_hf*n_hf] * n_hl + [n_hf])

        model_size_byte = (sum(wgts_list) + sum(biases_list)) * (1 - prune_rate) * 32 / 8
        return model_size_byte
    data_trains_cp_df["size_byte"] = list(map(get_model_size_byte, data_trains_cp_df.values))
    pass


def create_enanched_datasets_train(data_trains_df:pd.DataFrame, conf_dict: dict, timestamp_file = None) -> pd.DataFrame:
    """Add models' size for each model trained with input dataset.
    Args:
    -----
    `data_trains_cp_df` - input dataframe to be updated.\n
    `conf_dict` - dict object, with constraints about how carry out updating input dataset.\n
    `timestamp_file` - time stamp to be used for naming output dataset.\n
    Returns:
    --------
    `pd.DataFrame` - enanched output dataset improved with more details.\n
    """
    data_trains_cp_df = copy.deepcopy(data_trains_df)

    total_steps = 4
    with tqdm.tqdm(total=total_steps) as pbar:
        pbar.write("add_cmd_line_options...")
        data_trains_cp_df = add_cmd_line_options(data_trains_cp_df)
        pbar.update(1)

        pbar.write("add_prune_class...")
        add_prune_class(data_trains_cp_df)
        pbar.update(1)

        pbar.write("add_prune_rate...")
        add_prune_rate(data_trains_cp_df)
        pbar.update(1)

        pbar.write("add_models_size_byte...")
        add_models_size_byte(data_trains_cp_df)
        pbar.update(1)

        data_trains_cp_df["footprint"] = 1 - data_trains_cp_df["prune_rate"].values
        w, h = conf_dict["w"], conf_dict["h"]
        data_trains_cp_df["bpp"] = data_trains_cp_df["size_byte"] * 8 / (w * h)
        data_trains_cp_df["CR"] = w * h / data_trains_cp_df["size_byte"]
        pass

    if timestamp_file is None:
        timestamp_file = time.time()
        pass
    dest_dir_csv = os.path.join(conf_dict["root_dest_results"], f"pool_{timestamp_file}")
    dest_file_path_csv = os.path.join(dest_dir_csv, f"enanched_pool_{timestamp_file}.csv")

    if not os.path.isdir(dest_dir_csv):
        try:
            os.makedirs(dest_dir_csv)
            print(f"Dest directory '{dest_dir_csv}' created!")
            pass
        except:
            print(f"Dest directory '{dest_dir_csv}' already exists!")
            pass
    else: 
        print(f"Dest directory '{dest_dir_csv}' already exists!")
        pass
    data_trains_cp_df.to_csv(dest_file_path_csv)
    show_columns = "date_train,prune_techs,prune_rate,size_byte,footprint,bpp,CR".split(",")
    targets = "sidelength,n_hf,n_hl,num_epochs,lr,lambda_L_1,lambda_L_2".split(",")
    print(data_trains_cp_df[show_columns + targets].head(5))

    print(data_trains_cp_df[show_columns + targets].info())
    return dest_file_path_csv, data_trains_cp_df
