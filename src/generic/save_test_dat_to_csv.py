from pprint import pprint
import os
import sys
import pandas as pd
import yaml


def get_a_record() -> dict:
    """TODO COMMENT IT."""

    '''
    dataset_columns_time = "experiment_date,date_train,date_test,init_from,root_dir,model_name"
    dataset_columns_scores = "size_byte,footprint,psnr,bpp,CR,mse,ssim,time,entropy"
    dataset_columns_settings = "scheduler_name,scheduler,prune_techs,prune_rate,quant_techs,command_line,num_epochs,n_hl,n_hf,w,h,L1,L2"

    dataset_columns = []
    dataset_columns += dataset_columns_time.split(",")
    dataset_columns += dataset_columns_scores.split(",")
    dataset_columns += dataset_columns_settings.split(",")
    '''

    dataset_columns = "date_train,date_test,mse,psnr,ssim,time,size_byte".split(",")
    tmp_record = dict(zip(dataset_columns, ["-"] * len(dataset_columns)))
    return tmp_record


def add_to_dataset(file_name: str, columns: list, a_record: list) -> pd.DataFrame:
    """TODO COMMENT IT."""

    if os.path.exists(file_name) is False:
        df = pd.DataFrame(data=[a_record], columns = columns)
    else:
        df = pd.read_csv(file_name)
        if 'Unnamed 0' in df.columns:
            df = df.drop(['Unnamed 0'], axis=1)
        tmp_df = pd.DataFrame(data=[a_record], columns = columns)
        df = df.append(tmp_df)
        pass
    return df


def write_to_csv(file_name: str, df: pd.DataFrame) -> None:
    """TODO COMMENT IT."""
    dir_name = os.path.dirname(file_name)
    try:
        os.makedirs(dir_name)
    except: pass
    df.to_csv(file_name, index=False)
    pass


def create_record_from_app(opt, results_test, app = None, args = None, logdir = None):
    """TODO COMMENT IT."""

    a_record = get_a_record()
    try:
        df_model = app.get_dataframe_model()

        net_layers = list(df_model["Name"].values)
        net_layers = net_layers[0:(len(net_layers)-1)]
        wts_sparse = list(df_model["NNZ (sparse)"].values)
        wts_sparse = wts_sparse[:(len(wts_sparse)-1)]
        # pprint(wts_sparse)

        date_train = None
        if app.logdir:
            print(app.logdir)
            tmp_log_dir = os.path.normpath(app.logdir)
            date_train = os.path.basename(tmp_log_dir)
        pass
    except:
        print(logdir)
        tmp_log_dir = os.path.normpath(logdir)
        date_train = os.path.basename(tmp_log_dir)
        pass
    
    columns = "date_train,date_test,mse,psnr,ssim,time".split(",")
    a_record_vals = [date_train, date_train] + list(results_test)
    for k, v in zip(columns, a_record_vals):
        a_record[k] = v
        pass
    try:
        with open(opt.compress) as compress_file:
            compress_dict = yaml.load(compress_file, Loader=yaml.FullLoader)
            # pprint(compress_dict)
        if "quantizers" in compress_dict.keys():
            keys = list(compress_dict["quantizers"]["linear_quantizer"].keys())
            # pprint(keys)
            values = list(compress_dict["quantizers"]["linear_quantizer"].values())
            # pprint(values)
            keys_p = list(compress_dict["policies"][0].keys())[1:]
            # pprint(keys_p)
            values_p = list(compress_dict["policies"][0].values())[1:]
            # pprint(values_p)

            # columns = "date,MSE,PSNR,SSIM,TIME".split(",") + keys + keys_p + net_layers
            # a_record = [date_train] + list(results_test) + values + values_p + wts_sparse
            pass
    except:
        pass
    return a_record


def save_test_data_to_csv(opt, results_test, app = None, args = None, logdir = None):
    """TODO COMMENT IT."""
    file_name = opt.save_test_data_to_csv_path

    if app:
        columns = "date_train,date_test,mse,psnr,ssim,time".split(",")
        a_record = create_record_from_app(opt, results_test, app = None, args = None, logdir = None)
        pass

    columns = list(a_record.keys())
    a_record = list(a_record.values())
    df = add_to_dataset(file_name, columns = columns, a_record = a_record)
    write_to_csv(file_name, df)
    pass
