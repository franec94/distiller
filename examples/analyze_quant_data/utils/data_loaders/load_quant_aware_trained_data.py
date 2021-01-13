from utils.libs.std_python_libs import *
from utils.libs.data_science_libs import *
from utils.libs.graphics_and_interactive_libs import *


def load_siren_quant_aware_spec_df(net_layers):
    path_data_spec_quant_aware = 'C:\\Users\\Francesco\\Desktop\\quant_spec_siren64_5.txt'
    headers = []
    for a_layer in net_layers:
        headers.extend([f"{a_layer}-{a_feature}" for a_feature in "ws,act,bias".split(",")])
    siren_quant_aware_spec_df = pd.read_csv(path_data_spec_quant_aware, sep="\t", names=headers)
    # siren_quant_aware_spec_df.head(5)
    return siren_quant_aware_spec_df

def load_spec_df(net_layers, w, h, cropped_file_size_bits):
    siren_quant_aware_spec_df = load_siren_quant_aware_spec_df(net_layers)
    n = siren_quant_aware_spec_df.shape[0]
    odd_rows = siren_quant_aware_spec_df.iloc[range(0, n, 2),:]
    even_rows = siren_quant_aware_spec_df.iloc[range(1, n, 2),:]

    record_list = []
    for items, items_2 in zip(odd_rows.values, even_rows.values):
        items = np.array(list(map(int, items)))
        items_2 = np.array(list(map(int, items_2)))
        model_size_byte = np.sum(items * items_2) / 8
        bpp = model_size_byte * 8 / (w * h)
        CR = cropped_file_size_bits / np.sum(items * items_2)
    
        a_record = [model_size_byte, bpp, CR]
        record_list.append(a_record)
        pass
    spec_df = pd.DataFrame(record_list, columns = "model_size_byte,bpp,CR".split(","))
    # spec_df.head(5)
    return spec_df

def load_siren_quant_aware_data_df(net_layers, w, h, cropped_file_size_bits):
    spec_df = load_spec_df(net_layers, w, h, cropped_file_size_bits)
    path_data = 'C:\\Users\\Francesco\\Desktop\\quant_aware_train_data_siren64_5.csv'
    siren_quant_aware_data_df = pd.read_csv(path_data)
    # siren_quant_aware_data_df.head(5)
    # siren_quant_aware_data_df.info()
    # siren_quant_aware_data_df[["MSE","PSNR","TIME"]].describe()
    
    nrows = siren_quant_aware_data_df.shape[0]
    siren_quant_aware_df = pd.concat([siren_quant_aware_data_df.iloc[0:nrows,:], spec_df.iloc[0:nrows,:]], axis = 1)
    # siren_quant_aware_df.head(5)
    return siren_quant_aware_df

def load_siren_quant_aware_data_df_v2():
    path_data = 'C:\\Users\\Francesco\\Desktop\\quant_aware_train_data_siren64_5.csv'
    df = pd.read_csv(path_data)
    return
