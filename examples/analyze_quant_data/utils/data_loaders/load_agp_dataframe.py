from utils.libs.std_python_libs import *
from utils.libs.data_science_libs import *
from utils.libs.graphics_and_interactive_libs import *

root_filespaths = "C:\\Users\\Francesco\\Documents\\thesys\\code\\local_projects\\siren-train-logs\\notebooks\\analyze_a_run"
# agp_prune_rate_filepath = os.path.join(root_filespaths, "agp-pruning-rate-siren_65_5.txt")
# agp_prune_scores_filepath = os.path.join(root_filespaths, "agp-pruning-scores-siren_65_5.txt")

agp_prune_rate_filepath = os.path.join(root_filespaths, "agp_prune_rate_test.txt")
agp_prune_scores_filepath = os.path.join(root_filespaths, "test_agp_data.txt")

# columns_prune_rate = "net.0.linear,net.1.linear,net.2.linear,net.3.linear,net.4.linear,net.5.linear,net.6".split(",")
columns_prune_rate = "Date	Prune Tech.	Freq. Updated	Target Sparsity (%)	Achieved Sparsity (%)	net.0.linear	Sparsity (%)	net.1.linear	Sparsity (%)	net.2.linear	Sparsity (%)	net.3.linear	Sparsity (%)	net.4.linear	Sparsity (%)	net.5.linear	Sparsity (%)	net.6	Sparsity (%)	Footprint (Byte)	Footprint (%)".split("\t")


# columns_prune_scores = "mse,psnr,ssim".split(",")
columns_prune_scores = "Date,Prune Tech.,Freq. Update (epochs), Target Sparsity (%), Achieved Sparsity (%), Psnr (db), Bpp, Footprint (Byte), Footprint (%), Img Width	,Img Height, Mse, Ssim (%)".split(",")
columns_prune_scores = list(map(lambda xx: xx.strip(), columns_prune_scores))


def read_txt_file(text_filepath, columns, skip_elems_pos = []):
    """Load dataframe with data from input .txt file.
    Args:
    -----
    `text_filepath` - str object, file path to local file system where data are stored within plain .txt file.\n
    `columns` - list object, dataframe target columns.\n
    Return:
    -------
    `data_df` - pd.DataFrame object.\n
    `rows` - python list object with data from which dataframe has been created.\n
    """
    rows = None
    
    # Protocol process txt file
    filter_out_empyt_rows = lambda a_row: len(a_row) != 0
    def map_row_to_list(a_row, skip_elems_pos=skip_elems_pos):
        elms = None
        try:
            a_row_tmp = a_row.strip()
            a_row_tmp = re.sub("( ){1,}", " ", a_row_tmp)
            a_row_tmp = a_row_tmp.replace("\t", " ")
            elms = a_row_tmp.split(" ")
            # print(elms)
            
            if len(elms) == 0: return []
            if len(elms) != len(columns): return []
            
            replace_comma_with_dot = lambda a_elm: re.sub(",", ".", a_elm) if  "," in a_elm else a_elm
            elms = list(map(replace_comma_with_dot, elms))
            # print(elms)
            
            remove_percetage = lambda a_elm: re.sub("%", "", a_elm) if  "%" in a_elm else a_elm
            elms = list(map(remove_percetage, elms))
            # print(elms)
            
            def map_to_float_some(item, skip_elems_pos=skip_elems_pos):
                pos, val = item
                if skip_elems_pos != [] and pos in skip_elems_pos: return val
                return float(val)
            elms = list(map(map_to_float_some, enumerate(elms)))
            # print(elms)
        except Exception as err:
            print(err)
            return []
        return elms
    filter_out_empyt_lists = lambda a_list: a_list != []

    # Read raw data from .txt file
    with open(text_filepath, "r") as f:
        rows = f.read().split("\n")
        filename = os.path.basename(text_filepath)
        print(f"Read from {filename} # rows: {len(rows)}")

    # pprint(rows)
    # Apply Protocol to process txt file
    rows = list(filter(filter_out_empyt_rows, rows))
    # pprint(rows)
    rows = list(map(map_row_to_list, rows))
    # pprint(rows)
    rows = list(filter(filter_out_empyt_lists, rows))
    # pprint(rows)

    data_df = pd.DataFrame(data=rows, columns=columns)
    return data_df, rows

def load_agp_dataframe():
    """Load dataframe with data and scores about AGP-aware pruning.
    Return:
    -------
    `agp_df` - pd.DataFrame object.\n
    """
    agp_df = None
    data_pr_df, data_ps_df = None, None

    try:
        data_pr_df, rows_pr = read_txt_file(text_filepath=agp_prune_rate_filepath, columns=columns_prune_rate, skip_elems_pos=[0,1])
    except Exception as err:
        print(f"An error occurs when processing: {agp_prune_rate_filepath}")
        print(f"{str(err)}")
        pass
    try:
        data_ps_df, rows_ps = read_txt_file(text_filepath=agp_prune_scores_filepath, columns=columns_prune_scores, skip_elems_pos=[0,1])
        pass
    except Exception as err:
        print(f"An error occurs when processing: {agp_prune_scores_filepath}")
        print(f"{str(err)}")
        pass

    # joined_columns = list(data_pr_df.columns) + list(data_ps_df.columns)
    # agp_df = pd.concat([data_pr_df, data_ps_df], axis=1, names=joined_columns)
    
    # return agp_df
    return data_pr_df, data_ps_df
