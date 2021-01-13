
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


def calculate_uniformed_jpeg_csv_out_file(jpeg_uniform_df: pd.DataFrame, conf_dict: dict) -> pd.DataFrame:
    """Compute, save and retrieve updated and uniformed dataframe version of input provided dataframe.
    Args:
    -----
    `jpeg_uniform_df` - pd.DataFrame.\n
    `conf_dict` - dict object with conf data.\n
    Return:
    -------
    `jpeg_uniform_df` -  pd.DataFrame, updated instance.\n
    """
    baseline_model_size_bit = get_model_size_in_bit(conf_dict)

    rename_columns = "file_size_bits".split(",")
    renamed_columns = "size(byte)".split(",")
    for old_col_name, new_col_name in zip(rename_columns, renamed_columns):
        if old_col_name not in jpeg_uniform_df.columns: continue
        jpeg_uniform_df[f"{new_col_name}"] = jpeg_uniform_df[f"{old_col_name}"].values
        jpeg_uniform_df = jpeg_uniform_df.drop([f"{old_col_name}"], axis = 1)
        pass

    jpeg_uniform_df["size(byte)"] = jpeg_uniform_df["size(byte)"] / 8
    jpeg_uniform_df["footprint(%)"] = jpeg_uniform_df["size(byte)"] * 8 / baseline_model_size_bit * 100

    def create_compression_class_column(item):
        compression, quality = item
        return f"{compression.upper()}:{int(quality)}"

    drop_columns = "quality,compression,width,heigth".split(",")
    jpeg_uniform_df["cmprss-class"] = list(map(create_compression_class_column, jpeg_uniform_df[[f"compression", f"quality"]].values))
    for old_col_name in drop_columns:
        if old_col_name not in jpeg_uniform_df.columns: continue
        jpeg_uniform_df = jpeg_uniform_df.drop([f"{old_col_name}"], axis = 1)
        pass

    uniform_csv_dir = conf_dict['dest_dir']
    if not os.path.isdir(uniform_csv_dir):
        try: os.makedirs(uniform_csv_dir)
        except:pass
        pass

    jpeg_uniform_df['date'] = ['-'] * jpeg_uniform_df.shape[0]
    jpeg_uniform_df['init-from'] = ['-'] * jpeg_uniform_df.shape[0]
    
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
    jpeg_uniform_df["cmprss-class-2"] = list(map(create_compression_class_2, jpeg_uniform_df[f"cmprss-class"].values))

    columns_order = "date,init-from,size(byte),footprint(%),psnr,bpp,CR,mse,ssim,cmprss-class,cmprss-class-2".split(",")
    jpeg_uniform_df = jpeg_uniform_df[columns_order]
    
    jpeg_uniform_uniform_csv_path = os.path.join(f"{uniform_csv_dir}", "jpeg_uniform.csv")
    jpeg_uniform_df.to_csv(jpeg_uniform_uniform_csv_path)
    # jpeg_uniform_df.head(5)
    print(f"Results saved to: {jpeg_uniform_uniform_csv_path}")
    return jpeg_uniform_df


def main(args):

    print('Check conf file exists...')
    check_file_exists(args.input_conf_file)
    print('Read conf content...')
    conf_dict = read_input_conf_file(args.input_conf_file)
    # pprint(conf_dict)

    print('Load input image.')
    im = load_target_image(image_file_path = conf_dict["image_file_path"])
    print('Image Name:', conf_dict["image_file_path"] if conf_dict["image_file_path"] else "Cameramen")
    print('Image size:', im.size)

    print('Crop input image.')
    crop_size = (conf_dict["w"],conf_dict["h"])
    im_cropped = get_cropped_by_center_image(im, target = crop_size)
    cropped_file_size_bits = None
    with BytesIO() as f:
        im_cropped.save(f, format='PNG')
        cropped_file_size_bits = f.getbuffer().nbytes * 8
        pass
    
    print('Calculate several trials of jpeg image processing method...')
    result_jpeg_df = load_jpeg_baseline(im_cropped)
    jpeg_uniform_df = copy.deepcopy(result_jpeg_df)
    
    print('Save results...')
    jpeg_uniform_df = calculate_uniformed_jpeg_csv_out_file(jpeg_uniform_df, conf_dict)
    print(jpeg_uniform_df.head(3))
    
    if conf_dict['save_to_cmprs_dataset']:
        
        uniform_csv_dir = conf_dict['dest_dir']
        check_dir_exists(uniform_csv_dir)

        cmprs_dataset_name = conf_dict['cmprs_dataset_name']
        cmprs_df_csv_path = os.path.join(f"{uniform_csv_dir}", f"{cmprs_dataset_name}")   

        print("Appending structured data in:")
        print(f"\t{cmprs_df_csv_path}")
        if not check_file_exists(cmprs_df_csv_path, False):
            print(f"Created and not updated dataset file named: {cmprs_df_csv_path}")
            cmprs_df = copy.deepcopy(jpeg_uniform_df)
            cmprs_df.to_csv(cmprs_df_csv_path)
        else:
            print(f"Updating dataset file named: {cmprs_df_csv_path}...")
            cmprs_df = pd.read_csv(cmprs_df_csv_path)
            if "Unnamed: 0" in cmprs_df.columns:
                cmprs_df = cmprs_df.drop(["Unnamed: 0"], axis = 1)
        
            print('Before:')
            print('CMPRS DF:', cmprs_df.shape)
            print('RAW DF:', jpeg_uniform_df.shape)
            
            list_dfs = [cmprs_df, jpeg_uniform_df]
            cmprs_df = pd.concat(list_dfs, axis = 0, ignore_index=True)
            cmprs_df.to_csv(cmprs_df_csv_path)
            
            print('After:')
            print(cmprs_df.shape)
            print(cmprs_df.head(5))
        pass
    pass


if __name__ == "__main__":
    print('Parse input args...')
    args = parser.parse_args()
    main(args)
    pass
