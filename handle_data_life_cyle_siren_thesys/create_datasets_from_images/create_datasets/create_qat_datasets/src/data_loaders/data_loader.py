from src.libs.std_python_libs import *

import numpy as np
import pandas as pd

# utils.handle_server_connections.work 
# ----------------------------------------------- #
from src.handle_server_connections.work import fetch_data
from src.handle_server_connections.work import fetch_data_by_status
from src.handle_server_connections.work import fetch_data_by_constraints
from src.handle_server_connections.work import get_info_from_logged_parser
from src.handle_server_connections.work import insert_data_read_from_logs

# functiions - imports # from src.utils.functions
# ----------------------------------------------- #
from src.utils.functions import read_conf_file
from src.utils.functions import load_target_image
from src.utils.functions import get_dict_dataframes
from src.utils.functions import get_dataframe

from src.handle_dataframes import prepare_and_merge_target_dfs, calculate_several_jpeg_compression, get_cropped_by_center_image

def load_data_baseline(im_cropped, conf_file_path = 'conf.txt'):
    conf_data = read_conf_file(conf_file_path)
    
    cropped_file_size_bits = None
    with BytesIO() as f:
        im_cropped.save(f, format='PNG')
        cropped_file_size_bits = f.getbuffer().nbytes * 8
        pass
    
    typename = 'QueryConstraints'
    field_names = "image;date;timestamp;hidden_features;image_size;status".split(";")
    field_types = "str;str;str;int;str;str".split(";")

    QueryConstraints = collections.namedtuple(typename, field_names)
    
    image = dict(zip(['type', 'val'], [str, list(sorted("cameramen".split(";")))]))
    date = None
    timestamp = dict(zip(['type', 'val'], [str, list(sorted("".split(";")))]))
    hidden_features = dict(zip(['type', 'val'], [int, list(sorted("".split(";")))]))
    image_size = dict(zip(['type', 'val'], [str, list(sorted("[256,256]".split(";")))]))
    status = dict(zip(['type', 'val'], [str, list(sorted("done".split(";")))]))

    fields_list = [image, date, timestamp, hidden_features, image_size, status]
    constraints = QueryConstraints._make(fields_list)

    records_list, result_dict_df, query_str, chained_constraints = fetch_data_by_constraints(
        conf_data, constraints, fetch_data_downloaded = True)
    
    data = list(map(operator.itemgetter(1), result_dict_df.items()))
    train_df = pd.concat(data)
    train_df['file_size_bits'] = train_df['#params'].values * 32
    train_df['CR'] = cropped_file_size_bits / (train_df['#params'].values * 32)
    train_df['bpp'] = train_df['#params'].values * 32 / (im_cropped.size[0] * im_cropped.size[1])
    train_df['compression'] = list(map(lambda hf: f'siren-{hf}', train_df['hf'].values.astype(dtype = np.int)))
    return train_df

def load_jpeg_baseline(im_cropped):
    # --- Run several trials for JPEG compression.
    # --- Array of qualities to be tested in compression.
    qualities_arr = np.arange(20, 95+1, dtype = np.int)
    cropped_file_size_bits = None
    with BytesIO() as f:
        im_cropped.save(f, format='PNG')
        cropped_file_size_bits = f.getbuffer().nbytes * 8
        pass

    result_tuples, failure_qualities = \
      calculate_several_jpeg_compression(im_cropped, cropped_file_size_bits, qualities_arr)

    # data = list(map(lambda xx: xx._asdict(), result_tuples))
    data = list(map(operator.methodcaller('_asdict'), result_tuples))
    result_jpeg_df = pd.DataFrame(data = data)
    result_jpeg_df['compression'] = ['jpeg'] * result_jpeg_df.shape[0]
    return result_jpeg_df