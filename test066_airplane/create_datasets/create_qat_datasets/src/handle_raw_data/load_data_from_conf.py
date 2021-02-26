from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from src.utils.functions import *
from src.end_to_end_utils.end_to_end_utils import *

import PyPDF2
import img2pdf
import fitz

def save_all_images_as_merged_pdf(figures_list: list , args = None):
    """TODO comment .it"""

    if args:
        pdf_filename =  os.path.join(
            args.output_dir_path, "merged.pdf")
    else:
        pdf_filename =  os.path.join("merged.pdf")
        pass
    
    doc = fitz.open()                            # PDF with the pictures
    for i, f in enumerate(figures_list):
        img = fitz.open(f) # open pic as document
        rect = img[0].rect                       # pic dimension
        pdfbytes = img.convertToPDF()            # make a PDF stream
        img.close()                              # no longer needed
        imgPDF = fitz.open("pdf", pdfbytes)      # open stream as PDF
        page = doc.newPage(width = rect.width,   # new page with ...
                           height = rect.height) # pic dimension
        page.showPDFpage(rect, imgPDF, 0) 
               # image fills the page
    doc.save(pdf_filename)
    pass

# ---------------------------------------------- #
# Fetch Jpeg+Baseline and Image Data
# ---------------------------------------------- #

def get_agp_pruned_model(pruned_df: pd.DataFrame, conf_data_dict: dict) -> pd.DataFrame:

    timestamp: str = conf_data_dict["timestamp"]
    
    pos = pruned_df["date"] == timestamp
    pruned_model_df: pd.DataFrame = pruned_df[pos].head(1)
    return pruned_model_df


def get_teacher_model(baseline_df: pd.DataFrame, unique_pairs: list, n_hf_t: int, n_hl_t: int) -> pd.DataFrame:
    """TODO comment .it"""
    targtes_dfs_list: list = []
    for a_pair in unique_pairs:
        n_hf, n_hl = a_pair
        if n_hf != n_hf_t: continue
        if n_hl != n_hl_t: continue

        pos = baseline_df["cmprss-class"].str.contains(f'{n_hf}')
        pos2 = baseline_df["cmprss-class"].str.contains(f'{n_hl}')
        a_sub_df = baseline_df[(pos) & (pos2)]
        if a_sub_df.shape[0] != 0:
            targtes_dfs_list.append(a_sub_df.head(1))
        pass
    if targtes_dfs_list == []: return pd.DataFrame()
    teachers_bslns_df = pd.concat(targtes_dfs_list, axis = 0, ignore_index = True)

    return teachers_bslns_df


def get_reference_datasets():
    """TODO comment .it"""
    baseline_df, cmprss_df = load_full_cmprss_dataset()

    jpeg_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Jpeg".upper()]
    siren_bsln_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Siren".upper()]
    agp_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Agp".upper()]

    jpeg_df["prune_rate_intervals"] = ["JPEG"] * jpeg_df.shape[0]
    siren_bsln_df["prune_rate_intervals"] = ["SIREN"] * siren_bsln_df.shape[0]
    return jpeg_df, siren_bsln_df, baseline_df, agp_df


def get_target_images():
    """TODO comment .it"""
    camera = load_target_image()
    camera_crop = get_cropped_by_center_image(im=camera)
    camera_crop
    size_byte_crop = sys.getsizeof(camera_crop.tobytes())
    return camera, camera_crop, size_byte_crop


# ---------------------------------------------- #
# handle dataset
# ---------------------------------------------- #

def wrapper_calculate_size_byte(conf_data: dict, a_df: pd.DataFrame):
    """TODO comment .it"""
    cols = list(a_df.columns)
    ws_pruned = np.array(conf_data["raw_data"]["model_infos"]["ws_pruned"])
    biases = np.array(conf_data["raw_data"]["model_infos"]["biases"])
    def calculate_size_byte (
            a_row, cols=cols,
            ws_pruned=ws_pruned, biases=biases):
        a_row = pd.Series(a_row, index=cols)
        col = "scheduler"
        # if col not in a_row.index: raise Exception(f"{str(a_row)}")
        scheduler = a_row[col]
        if type(scheduler) == str:
            if scheduler == "-": return a_row["size_byte"]
            scheduler = eval(scheduler)
        # pprint(scheduler)
        overrides = scheduler["quantizers"]["linear_quantizer"]["overrides"]
        ws_quant = np.ones(len(ws_pruned)) * 32
        biases_quant = np.ones(len(ws_pruned)) * 32
        zero_point = np.zeros(len(ws_pruned))
        shift_scaler = np.zeros(len(ws_pruned))
        for k, v in overrides.items():
            k_pos = int(k.split(".")[1])
            if v["bits_weights"]:
                ws_quant[k_pos] = v["bits_weights"]
                shift_scaler[k_pos] = 32
                zero_point[k_pos] = v["bits_weights"]
            if v["bits_bias"]:
                biases_quant[k_pos] = v["bits_bias"]
            pass
        model_size_byte = np.sum(
            ws_quant * ws_pruned + \
            biases_quant * biases + \
            zero_point + shift_scaler \
        ) / 8
        return model_size_byte
    a_df["size_byte_th"] = list(map(calculate_size_byte, a_df.values))

    # a_df[show_cols].head(5)
    pass


def wrapper_calculate_CR(conf_data: dict, a_df: pd.DataFrame, a_image_byte_size: int):
    """TODO comment .it"""
    cols = list(a_df.columns)
    def calculate_CR(a_row, cols=cols, a_image_byte_size=a_image_byte_size):
        a_row = pd.Series(a_row, index=cols)
        col = "size_byte_th"
        model_size_byte = a_row[col]
        # print(model_size_byte)
        return a_image_byte_size / model_size_byte
    a_df["CR"] = list(map(calculate_CR, a_df.values))

    # a_df[show_cols].head(5)
    pass


def wrappr_calculate_bpp(conf_data: dict, a_df: pd.DataFrame):
    """TODO comment .it"""
    cols = list(a_df.columns)
    def calculate_bpp(a_row, cols=cols):
        a_row = pd.Series(a_row, index=cols)
        # pprint(a_row)
        cols = "h,w,size_byte_th".split(",")
        h, w, size_byte = a_row[cols]
        return size_byte * 8 / (h * w)
    a_df["bpp"] = list(map(calculate_bpp, a_df.values))

    # a_df[show_cols].head(5)
    pass


def wrapper_calculate_quant_techs(conf_data: dict, a_df: pd.DataFrame):
    """TODO comment .it"""
    cols = list(a_df.columns)
    def  calculate_quant_tech(a_row, cols=cols):
        a_row = pd.Series(a_row, index=cols)
        col = "scheduler"
        scheduler = a_row[col]
        if type(scheduler) == str:
            if scheduler == "-": return "-"
            scheduler = eval(scheduler)
        # pprint(scheduler)
        aqt = scheduler["quantizers"]["linear_quantizer"]["class"]
        aqt = ''.join(list(filter(lambda ch: ch.upper() == ch, list(aqt))))
        
        pcw = scheduler["quantizers"]["linear_quantizer"]["per_channel_wts"]
        if pcw: pcw = 'PCW'
        else: pcw = 'NNPCW'
        
        get_1st_ch = lambda x: x[0].upper()
        mode = scheduler["quantizers"]["linear_quantizer"]["mode"]
        mode = ''.join(list(map(get_1st_ch, mode.split("_"))))
        return f"{aqt}:{mode}:{pcw}"
    a_df["quant_techs"] = list(map(calculate_quant_tech, a_df.values))

    # a_df[show_cols].head(5)
    pass


def load_rawdata_from_conf(conf_data: dict, verbose: int = 0):
    """TODO comment .it"""

    success_readings: list = []
    failure_readings: list = []
    camera, camera_crop, size_byte_crop = \
        get_target_images()
    datasets_list = conf_data["raw_data"]["datasets_list"]


    if type(datasets_list) != list:
        datasets_list = [datasets_list]

    dfs_list: list = []
    total = len(datasets_list)
    with tqdm.tqdm(total = total) as pbar:
        for a_file in datasets_list:
            try:
                root_dir_exp = os.path.join(a_file)
                datasets_list: list = []
                for path in pathlib.Path(f"{root_dir_exp}").rglob('*.csv'):
                    # print(path.name)
                    # print(path)
                    datasets_list.append(path)
                    pass
                res_train_df = pd.read_csv(datasets_list[0])
                # pprint(res_train_df.columns)
                
                exp_info_df = pd.read_csv(datasets_list[1])
                # pprint(exp_info_df.columns)

                if len(res_train_df.columns) > len(exp_info_df.columns):
                    res_train_df, exp_info_df = exp_info_df, res_train_df

                a_df = copy.deepcopy(exp_info_df)
                if a_df.shape[0] != res_train_df.shape[0]:
                    a_df = a_df.iloc[:res_train_df.shape[0],:]
                    pass
                # show_cols = list(res_train_df.columns)
                for a_col in res_train_df.columns:
                    if a_col not in a_df.columns: continue
                    a_df[f"{a_col}"] = res_train_df[f"{a_col}"].values
                    pass
                # a_df[show_cols].head(5)

                if verbose == 1:
                    pbar.write('Res CSV -> %s %d' % (os.path.basename(datasets_list[0]), len(res_train_df.columns)))
                    pbar.write('Exp Infos CSV -> %s %d' % (os.path.basename(datasets_list[1]), len(exp_info_df.columns)))
                    # print('Res CSV ->', os.path.basename(datasets_list[0]), len(res_train_df.columns))
                    # print('Exp Infos CSV ->', os.path.basename(datasets_list[1]), len(exp_info_df.columns))
                    # pprint(res_train_df.columns)
                    # pprint(exp_info_df.columns)
                    pass

                wrapper_calculate_size_byte(conf_data, a_df)
                wrapper_calculate_CR(conf_data, a_df, size_byte_crop)
                wrappr_calculate_bpp(conf_data, a_df)
                wrapper_calculate_quant_techs(conf_data, a_df)

                # texp_info_res_merged_df = copy.deepcopy(a_df)
                dfs_list.append(a_df)
                success_readings.append(a_file)
            except Exception as err:
                # print(f"{str(err)}")
                pbar.write(f"{str(err)}")
                # print('Res CSV ->', datasets_list[0])
                # pprint(res_train_df.columns)
                # print('Exp Infos CSV ->', datasets_list[1])
                # pprint(exp_info_df.columns)
                # failure_readings.append(a_file)
                data_dict = dict(
                    err=f"{str(err)}",
                    res_csv_path=datasets_list[0],
                    info_train_path=datasets_list[1],
                )
                meta_tb = dict(tabular_data=data_dict.items())
                table_obj = tabulate.tabulate(**meta_tb)
                print(table_obj)
                pass
            pbar.update(1)
            pass
        pass

    if dfs_list == []:
        a_df = pd.DataFrame()
    else:
        a_df = pd.concat(dfs_list, axis = 0, ignore_index = True)
    
    return \
        (camera, camera_crop, size_byte_crop), \
        a_df, \
        (success_readings, failure_readings)
