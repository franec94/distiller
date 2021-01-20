from pprint import pprint
import os
import sys
import pandas as pd
import yaml
import tabulate
import torch


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


def add_a_row_to_dataframe(file_name: str, columns: list, a_record: list) -> pd.DataFrame:
    """TODO COMMENT IT."""

    if os.path.exists(file_name) is False:
        a_df = pd.DataFrame(data=[a_record], columns = columns)
    else:
        a_df = pd.read_csv(file_name)
        if 'Unnamed: 0' in a_df.columns:
            a_df = a_df.drop(['Unnamed: 0'], axis=1)
        if 'unnamed: 0' in a_df.columns:
            a_df = a_df.drop(['unnamed: 0'], axis=1)
        if 'Unnamed 0' in a_df.columns:
            a_df = a_df.drop(['Unnamed 0'], axis=1)
        tmp_df = pd.DataFrame(data=[a_record], columns = columns)
        a_df = a_df.append(tmp_df)
        pass
    return a_df


def write_dataframe_to_csv(file_name: str, a_df: pd.DataFrame) -> None:
    """TODO COMMENT IT."""
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass
    a_df.to_csv(file_name, index=False)
    pass


def create_record_from_app(opt, results_test, app = None, args = None, logdir = None):
    """TODO COMMENT IT."""

    a_record = get_a_record()
    date_train = None
    try:
        
        """
        df_model = app.get_dataframe_model()

        net_layers = list(df_model["Name"].values)
        net_layers = net_layers[0:(len(net_layers)-1)]
        wts_sparse = list(df_model["NNZ (sparse)"].values)
        wts_sparse = wts_sparse[:(len(wts_sparse)-1)]
        # pprint(wts_sparse)
        """

        if app.logdir:
            print(app.logdir)
            tmp_log_dir = os.path.normpath(app.logdir)
            date_train = os.path.basename(tmp_log_dir)
            pass
        pass
    except:
        # print(logdir)
        tmp_log_dir = os.path.normpath(logdir)
        date_train = os.path.basename(tmp_log_dir)
        pass
    
    columns = "date_train,date_test,mse,psnr,ssim,time".split(",")
    a_record_vals = [date_train, date_train] + list(results_test)
    for k, v in zip(columns, a_record_vals):
        a_record[k] = v
        pass
    """
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
    """
    return a_record


def save_test_data_to_csv(opt, results_test, app = None, args = None, logdir = None):
    """TODO COMMENT IT."""
    file_name = opt.save_test_data_to_csv_path

    if app:
        data_tb = dict(
            app_logdir=app.logdir,
            csv_file_path=file_name,
        )
        columns = "date_train,date_test,mse,psnr,ssim,time".split(",")
        model_size_byte = 0.0
        try:
            tmp_model_file = os.path.join(app.logdir, "tmp_model.pt")
            torch.save(app.model, tmp_model_file)
            model_size_byte = os.path.getsize(tmp_model_file)
        except Exception as err:
            raise err
            pass
        a_record = create_record_from_app(opt=opt, results_test=results_test, app = app, args = args, logdir = logdir)
        a_record["size_byte"] = model_size_byte
        for k, v in a_record.items():
            data_tb[k] = v
            pass

        columns = list(a_record.keys())
        a_record = list(a_record.values())
        a_df = add_a_row_to_dataframe(file_name=file_name, columns = columns, a_record = a_record)
        write_dataframe_to_csv(file_name, a_df)

        meta_tb = dict(
            tabular_data=data_tb.items()
        )
        a_table = tabulate.tabulate(**meta_tb)
        return a_table
    return None
