from pprint import pprint
import os
import sys
import pandas as pd
import yaml

def save_test_data_to_csv(opt, results_test, app):
    file_name = opt.save_test_data_to_csv_path
    df_model = app.get_dataframe_model()

    net_layers = list(df_model["Name"].values)
    net_layers = net_layers[0:(len(net_layers)-1)]
    # pprint(net_layers)
    # headers = []
    # for a_layer in net_layers: headers.extend([f"{a_layer}-{a_feature}" for a_feature in "ws,act,bias".split(",")])
    wts_sparse = list(df_model["NNZ (sparse)"].values)
    wts_sparse = wts_sparse[:(len(wts_sparse)-1)]
    # pprint(wts_sparse)

    date_train = None
    if app.logdir:
        print(app.logdir)
        tmp_log_dir = os.path.normpath(app.logdir)
        date_train = os.path.dirname(tmp_log_dir)

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

        columns = "date,MSE,PSNR,SSIM,TIME".split(",") + keys + keys_p + net_layers
        a_record = [date_train] + list(results_test) + values + values_p + wts_sparse
        if os.path.exists(file_name) is False:
            df = pd.DataFrame(data=[a_record], columns = columns)
        else:
            df = pd.read_csv(file_name).drop(['Unnamed 0'], axis=1)
            tmp_df = pd.DataFrame(data=[a_record], columns = columns)
            df = df.append(tmp_df)
        dir_name = os.path.dirname(file_name)
        try:
            os.makedirs(dir_name)
        except: pass
        df.to_csv(file_name)
    pass