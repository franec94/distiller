
from utils.libs import *


parser = argparse.ArgumentParser()
parser.add_argument("--input-conf-file", dest = "input_conf_file", type=str, required=True, \
    help="Source file path, whtinin local file system, to input conf options for running script."
)


def get_model_size_in_bit(conf_dict: dict) -> int:
    """Get model size in bit from conf data provided by conf dict object.
    Args:
    -----
    `conf_dict` - dict object with conf data.\n
    Return:
    -------
    `baseline_model_size_bit` - int value representing model size in bit.\n
    """
    baseline_values_ws = conf_dict["baseline_values_ws"].split(",")
    baseline_values_ws = list(map(float, baseline_values_ws))

    tot_ws_baseline = sum(baseline_values_ws)
    print('Tot number weigths', tot_ws_baseline)

    baseline_values_bias = conf_dict["baseline_values_bias"].split(",")
    baseline_values_bias = list(map(float, baseline_values_bias))

    tot_bias_baseline = sum(baseline_values_bias)
    print('Tot number biases', tot_bias_baseline)

    baseline_model_size_bit = (tot_bias_baseline + tot_ws_baseline) * 32
    print("Baseline(Byte)", baseline_model_size_bit / 8, "Baseline(bit)", baseline_model_size_bit )
    return int(baseline_model_size_bit)


def read_input_conf_file(conf_file_path: str, ext:str=".yaml") -> dict:
    """Read input conf file, which should be a .yaml file.
    Args:
    -----
    `conf_file_path` - str object to existing conf file. (Allowed=['.yaml', '.json'])\n
    Return:
    -------
    `dict` - object with configuratio options.\n
    Raise:
    ------
    `exception` - when no file is provided compliant with extention provided as input extention.\n
    """
    conf_dict = dict()
    allowed_ext = ['.yaml', '.json']

    _, file_ext = os.path.splitext(conf_file_path)
    if file_ext not in allowed_ext: raise Exception(f"Error: 'ext={ext}' not in {allowed_ext}")

    
    with open(conf_file_path, "r") as conf_fp:
        if file_ext == '.yaml':
            conf_dict = yaml.load(conf_fp, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError(f"File extentions '{file_ext}' not yet supported.")
    return conf_dict


def update_raw_df_attributes(raw_df: pd.DataFrame, conf_dict: dict) -> pd.DataFrame:
    """Update with some missing attributes raw dataframe.
    Args:
    -----
    `raw_df` - pd.Dataframe to be updated with som required attributes such as 'model_size_byte','CR','bpp'.\n
    `conf_dict` - dict python object with conf data to calculate missing attributes.\n
    Return:
    -------
    `raw_df` - pd.Dataframe, updated version.\n
    """

    w, h = conf_dict["w"], conf_dict["h"]
    n_hf, n_hl = conf_dict["n_hf"], conf_dict["n_hl"]
    arr_wts_no = np.array(list(map(int, conf_dict["arr_wts_no"].split(","))))
    columns = raw_df.columns
    def calculate_model_size(a_row, columns=columns, hl=n_hl, hf=n_hf, arr_wts_no = arr_wts_no):
        a_row_dict = dict(zip(columns, a_row))
        wts = [f"module.net.{ii}.linear.weight" for ii in range(0, n_hl+1)]
        wts_list = dict(filter(lambda item: item[0] in wts, a_row_dict.items()))
        a_row_dict[f"module.net.{hl+1}.weight"]
        wts_list[f"module.net.{hl+1}.weight"] = a_row_dict[f"module.net.{hl+1}.weight"]
        
        overrides = eval(a_row_dict["overrides"])
        arr_bits_weights = []
        arr_wts_no = []
        for k, v in overrides.items():
            bits_weights = v['bits_weights'] if v['bits_weights'] else 32
            arr_bits_weights.append(bits_weights)
            wts_no = wts_list[f"module.{k}.weight"]
            arr_wts_no.append(wts_no)
        
        arr_bits_weights = np.array(arr_bits_weights) # ;pprint(arr_bits_weights)
        # pprint(arr_bits_weights * arr_wts_no)
        model_size_byte = (32 * 2 + np.sum(arr_bits_weights * arr_wts_no) + 32 * hl * hf + 32) / 8
        # print("Model size byte:", model_size_byte)
        return model_size_byte

    raw_df['model_size_byte'] = list(map(calculate_model_size, raw_df.values[0:]))
    raw_df['bpp'] = list(map(lambda item: item * 8 / (w * h), raw_df["model_size_byte"].values))
    raw_df['CR'] = list(map(lambda item: (w * h * 8) / (item * 8), raw_df["model_size_byte"].values))
    return raw_df


def calculate_uniform_qat_data_out_csv(raw_uniform_df: pd.DataFrame, conf_dict: dict) -> pd.DataFrame:
    """Compute, save and retrieve updated and uniformed dataframe version of input provided dataframe.
    Args:
    -----
    `raw_uniform_df` - pd.DataFrame.\n
    `conf_dict` - dict object with conf data.\n
    Return:
    -------
    `raw_uniform_df` -  pd.DataFrame, updated instance.\n
    """
    baseline_model_size_bit = get_model_size_in_bit(conf_dict)
    kind_freq, target_date = conf_dict["kind_freq"], conf_dict["target_date"]

    rename_columns = "MSE,PSNR,SSIM,TIME,model_size_byte".split(",")
    renamed_columns = "mse,psnr,ssim,time,size(byte)".split(",")
    for old_col_name, new_col_name in zip(rename_columns, renamed_columns):
        if old_col_name not in raw_uniform_df.columns: continue
        raw_uniform_df[f"{new_col_name}"] = raw_uniform_df[f"{old_col_name}"].values
        raw_uniform_df = raw_uniform_df.drop([f"{old_col_name}"], axis = 1)
        pass

    raw_uniform_df["date"] = list(map(lambda item: item[3:], raw_uniform_df["date"].values))
    raw_uniform_df["footprint(%)"] = raw_uniform_df["size(byte)"] * 8 / baseline_model_size_bit * 100

    def create_compression_class_column(item, prune_tech = 'AGP'):
        class_attr, mode_attr, per_channel_wts_attr = item
        class_name, mode_name, pcw_name = None, None, 'NNPCW'
        if per_channel_wts_attr:
            pcw_name = 'PCW'
        if len(mode_attr.split("_")) > 1:
            mode_name = ''.join(list(map(lambda item: item[0].upper(), mode_attr.split("_"))))
        else:
            mode_name = mode_attr[0].upper()
        class_name = ''.join(list(filter(lambda letter: letter != letter.lower(), list(class_attr))))
        if prune_tech:
            return f"{prune_tech}+{class_name}:{mode_name}:{pcw_name}"
        return f"{class_name}:{mode_name}:{pcw_name}"

    drop_columns = "class,mode,per_channel_wts".split(",")
    raw_uniform_df["cmprss-class"] = list(map(create_compression_class_column, raw_uniform_df[drop_columns].values))
    for old_col_name in drop_columns:
        if old_col_name not in raw_uniform_df.columns: continue
        raw_uniform_df = raw_uniform_df.drop([f"{old_col_name}"], axis = 1)
        pass

    raw_uniform_df['init-from'] = [f'{target_date}'] * raw_uniform_df.shape[0]
    def create_compression_class_2(cmprss_class):
        if cmprss_class.startswith("jpeg".upper()):
            return cmprss_class.split(":")[0]
        if cmprss_class.startswith("agp".upper()):
            if len(cmprss_class.split("+")) == 1:
                return cmprss_class.split(":")[0]
        if cmprss_class.startswith("QATRLQ".upper()):
            pass
        if "QATRLQ".upper() in cmprss_class:
            return "QATRLQ"
        if cmprss_class.lower().startswith("Baseline".lower()):
            return "BASELINE" 
        return cmprss_class
    raw_uniform_df["cmprss-class-2"] = list(map(create_compression_class_2, raw_uniform_df[f"cmprss-class"].values))
    
    columns_order = "date,init-from,size(byte),footprint(%),psnr,bpp,CR,mse,ssim,cmprss-class,cmprss-class-2".split(",")
    raw_uniform_df = raw_uniform_df[columns_order]
    # raw_uniform_df.head(5)

    return raw_uniform_df


def get_raw_csv_path(conf_dict):
    kind_freq, target_date = conf_dict["kind_freq"], conf_dict["target_date"]
    input_dir = conf_dict["input_dir"]

     
    experimtent_timestamp = conf_dict["experimtent_timestamp"]
    if experimtent_timestamp:
        data_path_pieces = [
            f"{input_dir}",
            f"qat_experiment_from_agp_{target_date}_date_{experimtent_timestamp}",
            f"{kind_freq}-linear_quant_attempt_{target_date}.csv"
        ]
        conf_dict["raw_csv_path"] = os.path.join(*data_path_pieces)
        print("Looking for:", conf_dict["raw_csv_path"])
        if check_file_exists(conf_dict["raw_csv_path"], False): return
        pass
    
    conf_dict["raw_csv_path"] = os.path.join(f"{input_dir}", f"qat_experiment_from_agp_{target_date}", f"{kind_freq}-linear_quant_attempt_{target_date}.csv")
    print("Looking for:", conf_dict["raw_csv_path"])
    if check_file_exists(conf_dict["raw_csv_path"], False): return

    conf_dict["raw_csv_path"] = os.path.join(f"{input_dir}", f"{kind_freq}-linear_quant_attempt_{target_date}.csv")
    print("Looking for:", conf_dict["raw_csv_path"])
    if check_file_exists(conf_dict["raw_csv_path"], False): return

    raise Exception("Error input raw file path is not found!")


def main(args):

    print('Check conf file exists...')
    check_file_exists(args.input_conf_file)
    print('Read conf content...')
    conf_dict = read_input_conf_file(args.input_conf_file)
    pprint(conf_dict)

    print('Check raw csv file file exists...')
    kind_freq, target_date = conf_dict["kind_freq"], conf_dict["target_date"]
    input_dir = conf_dict["input_dir"]
    get_raw_csv_path(conf_dict)


    print('Load input image.')
    im = load_target_image(image_file_path = conf_dict["image_file_path"])
    print('Image Name:', conf_dict["image_file_path"] if conf_dict["image_file_path"] else "Cameramen")
    print('Image size:', im.size)

    print('Crop input image.')
    crop_size = (conf_dict["w"],conf_dict["h"])
    im_cropped = get_cropped_by_center_image(im, target = crop_size)
    print('Cropped Image size:', im_cropped.size)
    cropped_file_size_bits = None
    with BytesIO() as f:
        im_cropped.save(f, format='PNG')
        cropped_file_size_bits = f.getbuffer().nbytes * 8
        pass

    print('Load qat data to be uniformed...')
    raw_df = pd.read_csv(conf_dict["raw_csv_path"])
    raw_df = update_raw_df_attributes(raw_df, conf_dict)
    # print(raw_df.head(5))
    # sys.exit(0)

    print('Save results as uniformed csv structure...')
    raw_uniform_df = copy.deepcopy(raw_df)
    raw_uniform_df = calculate_uniform_qat_data_out_csv(raw_df, conf_dict)
    print(raw_uniform_df.head(3))
    print(raw_uniform_df.info())

    if conf_dict['flag_save_to_df_csv']:
        uniform_csv_dir = conf_dict['dest_dir']
        if not check_dir_exists(uniform_csv_dir, False):
            try: os.makedirs(uniform_csv_dir)
            except: pass
        kind_freq = conf_dict['kind_freq']
        target_date = conf_dict['target_date']

        raw_uniform_csv_path = os.path.join(f"{uniform_csv_dir}", f"{kind_freq}-siren_qatrlq_uniform_{target_date}.csv")
        print("Saving raw data to structured csv in:")
        print(f"\t{raw_uniform_csv_path}")
        # raw_uniform_df.to_csv(raw_uniform_csv_path)
        pass

    if conf_dict['flag_join_to_cmprs_df_csv']:
        
        uniform_csv_dir = conf_dict['dest_dir']
        check_dir_exists(uniform_csv_dir)

        cmprs_dataset_name = conf_dict['cmprs_dataset_name']
        cmprs_df_csv_path = os.path.join(f"{uniform_csv_dir}", f"{cmprs_dataset_name}")   

        print("Appending structured data in:")
        print(f"\t{cmprs_df_csv_path}")
        if not check_file_exists(cmprs_df_csv_path, False):
            print(f"Created and not updated dataset file named: {cmprs_df_csv_path}")
            cmprs_df = copy.deepcopy(raw_uniform_df)
            # cmprs_df.to_csv(cmprs_df_csv_path)
        else:
            print(f"Updating dataset file named: {cmprs_df_csv_path}...")
            cmprs_df = pd.read_csv(cmprs_df_csv_path)
            if "Unnamed: 0" in cmprs_df.columns:
                cmprs_df = cmprs_df.drop(["Unnamed: 0"], axis = 1)
        
            print('Before:')
            print('CMPRS DF:', cmprs_df.shape)
            print('RAW DF:', raw_uniform_df.shape)
            
            list_dfs = [cmprs_df, raw_uniform_df]
            cmprs_df = pd.concat(list_dfs, axis = 0, ignore_index=True)
            cmprs_df.to_csv(cmprs_df_csv_path)
            
            print('After:')
            print(cmprs_df.shape)
            print(cmprs_df.head(5))
        pass
    
    if 'flag_save_as_tmp' in conf_dict.keys() and conf_dict['flag_save_as_tmp']:
        uniform_csv_dir_tmp = os.path.join(conf_dict['dest_dir'], 'tmp')
        if not check_dir_exists(uniform_csv_dir_tmp, False):
            try: os.makedirs(uniform_csv_dir_tmp)
            except: pass
        kind_freq = conf_dict['kind_freq']
        target_date = conf_dict['target_date']

        raw_uniform_csv_path_tmp = os.path.join(f"{uniform_csv_dir_tmp}", f"{kind_freq}-siren_qatrlq_uniform_{target_date}.csv")
        print("Saving raw data to structured csv in:")
        print(f"\t{raw_uniform_csv_path_tmp}")
        raw_uniform_df.to_csv(raw_uniform_csv_path_tmp)
        pass

    pass


if __name__ == "__main__":
    print('Parse input args...')
    args = parser.parse_args()
    main(args)
    pass