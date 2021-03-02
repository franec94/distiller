from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from src.end_to_end_utils.create_static_graphics import create_static_scatter_graphic, create_static_scatter_graphic_with_boundaries
from src.end_to_end_utils.end_to_end_utils import *
from src.utils.functions import *

import PyPDF2
import img2pdf
import fitz


# ---------------------------------------------- #
# Fetch Jpeg+Baseline and Image Data
# ---------------------------------------------- #

def get_some_dfs():
    """TODO comment .it"""
    baseline_df, cmprss_df = load_full_cmprss_dataset()

    jpeg_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Jpeg".upper()]
    siren_bsln_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Siren".upper()]
    agp_df = cmprss_df[cmprss_df["cmprss-class-2"] == "Agp".upper()]

    jpeg_df["prune_rate_intervals"] = ["JPEG"] * jpeg_df.shape[0]
    siren_bsln_df["prune_rate_intervals"] = ["SIREN"] * siren_bsln_df.shape[0]
    return jpeg_df, siren_bsln_df, baseline_df, agp_df


def get_target_images() -> (PIL.Image, PIL.Image, int):
    """TODO comment .it
    Returns:
    -------
    `PIL.Image` - full image .\n
    `PIL.Image` -  cropped image.\n
    `int`.\n
    """
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
    ws_pruned = np.array(conf_data["exp_infos"]["model_infos"]["ws_pruned"])
    biases = np.array(conf_data["exp_infos"]["model_infos"]["biases"])
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


def merge_datasets_quant_data(conf_data: dict, verbose = 0):
    """TODO comment .it"""

    success_readings = []
    failure_readings = []
    camera, camera_crop, size_byte_crop = get_target_images()

    if type(conf_data["exp_infos"]["root_dir_exp"]) != list:
        conf_data["exp_infos"]["root_dir_exp"] = [conf_data["exp_infos"]["root_dir_exp"]]
        pass

    dfs_list = []
    for a_file in conf_data["exp_infos"]["root_dir_exp"]:

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
            # show_cols = list(res_train_df.columns)
            for a_col in res_train_df.columns:
                if a_col not in a_df.columns: continue
                a_df[f"{a_col}"] = res_train_df[f"{a_col}"].values
                pass
            # a_df[show_cols].head(5)

            if verbose == 1:
                print('Res CSV ->', os.path.basename(datasets_list[0]), len(res_train_df.columns))
                print('Exp Infos CSV ->', os.path.basename(datasets_list[1]), len(exp_info_df.columns))
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
            print(f"{str(err)}")
            print('Res CSV ->', datasets_list[0])
            pprint(res_train_df.columns)
            print('Exp Infos CSV ->', datasets_list[1])
            pprint(exp_info_df.columns)
            failure_readings.append(a_file)
            pass
        pass

    a_df = pd.concat(dfs_list, axis = 0, ignore_index = True)

    return camera, camera_crop, size_byte_crop, a_df, (success_readings, failure_readings)

# ---------------------------------------------- #
# Add Details To graphics
# ---------------------------------------------- #

def add_target_quality_range(a_df, tmp_jpeg_max_q, tmp_jpeg_min_q, conf_data, ax, a_x_key = 'bpp', a_y_key = 'psnr'):
    if tmp_jpeg_min_q.shape[0] == 0: return

    xmin, xmax = min(a_df[f"{a_x_key}"].values), max(a_df[f"{a_x_key}"].values)
    # ymin, ymax = min(a_df[f"{a_y_key}"].values), max(a_df[f"{a_y_key}"].values)
    
    # Add min quaility line threshold
    yval = tmp_jpeg_min_q[a_y_key].values[0]
    bpp = tmp_jpeg_min_q[a_x_key].values[0]
    q = tmp_jpeg_min_q['cmprss-class'].values[0].split(":")[1]
    hlines = dict(
        xmin=xmin, xmax=xmax, y=yval, linestyle = ':', color = 'green', alpha=0.7
    )
    ax.hlines(**hlines)
    ax.text(bpp, yval, f'jpeg({q})\npsnr={yval:.2f}\nbpp={bpp:.2f}', horizontalalignment='left',
        verticalalignment='center', )
    ax.scatter(bpp, yval, marker='^', color="orange", s=150, edgecolors='black')
    ax.scatter(bpp, yval, marker='.', color="red")

    # Add max quaility line threshold
    yval = tmp_jpeg_max_q[a_y_key].values[0]
    bpp = tmp_jpeg_max_q[a_x_key].values[0]
    q = tmp_jpeg_max_q['cmprss-class'].values[0].split(":")[1]
    hlines = dict(
        xmin=xmin, xmax=xmax, y=yval, linestyle = ':', color = 'green', alpha=0.7, label="jpeg: min/max w.r."
    )
    ax.hlines(**hlines)
    ax.text(bpp, yval, f'jpeg({q})\npsnr={yval:.2f}\nbpp={bpp:.2f}', horizontalalignment='left',
        verticalalignment='center',)
    # ax.scatter(bpp, yval, marker='^', color="orange", s=150, edgecolors='black', label="jpeg: min/max w.r.")
    ax.scatter(bpp, yval, marker='^', color="orange", s=150, edgecolors='black', )
    ax.scatter(bpp, yval, marker='.', color="red")
    pass


def add_some_data_examples_wrt_jpeg_wr(a_df, tmp_jpeg_max_q, tmp_jpeg_min_q, conf_data, ax, a_x_key = 'bpp', a_y_key = 'psnr', key_class = None, a_class = None):
    if key_class is None: return
    jmin_psnr = tmp_jpeg_min_q[a_y_key].values[0]
    jmax_psnr = tmp_jpeg_max_q[a_y_key].values[0]

    filter_data = (a_df[f"{key_class}"] == a_class) & (a_df[f"{a_y_key}"] >= jmin_psnr)
    a_df_min = a_df[filter_data].sort_values(by = [f"{a_y_key}"]).iloc[0,:]
    a_df_min = pd.DataFrame(data = [a_df_min.values], columns=a_df_min.index)

    ymax = max(a_df[f"{a_y_key}"].values)
    ymin = 0
    vlines = dict(
        ymin=ymin, ymax=ymax, x=8.0, linestyle = ':', color = 'black', alpha=0.7, label="hline/image bpp"
    )
    ax.vlines(**vlines)

    yval = a_df_min[a_y_key].values[0]
    bpp = a_df_min[a_x_key].values[0]
    color = "red" if bpp > 8 else "green"
    ax.text(bpp, yval, f'{a_class}\npsnr={yval:.2f}\nbpp={bpp:.2f}', horizontalalignment='left',
        verticalalignment='center',)
    # ax.scatter(bpp, yval, marker='^', color="orange", s=150, edgecolors='black', label="jpeg: min/max w.r.")
    ax.scatter(bpp, yval, marker='^', color=f"{color}", s=150, edgecolors='black', )

    filter_data = (a_df[f"{key_class}"] == a_class) & (a_df[f"{a_y_key}"] <= jmax_psnr)
    a_df_max = a_df[filter_data].sort_values(by = [f"{a_y_key}"], ascending = False).iloc[0,:]
    a_df_max = pd.DataFrame(data = [a_df_max.values], columns=a_df_max.index)

    yval = a_df_max[a_y_key].values[0]
    bpp = a_df_max[a_x_key].values[0]
    color = "red" if bpp > 8 else "green"
    ax.text(bpp, yval, f'{a_class}\npsnr={yval:.2f}\nbpp={bpp:.2f}', horizontalalignment='left',
        verticalalignment='center',)
    # ax.scatter(bpp, yval, marker='^', color="orange", s=150, edgecolors='black', label="jpeg: min/max w.r.")
    ax.scatter(bpp, yval, marker='^', color=f"{color}", s=150, edgecolors='black', )
    pass

# ---------------------------------------------- #
# Create Graphics
# ---------------------------------------------- #

def create_scatter_plot(a_df, conf_data, args = None, out_dir_path = None, a_target = None,
        teachers_bslns_df = pd.DataFrame(), ax = None,
        tmp_jpeg_max_q=pd.DataFrame(), tmp_jpeg_min_q=pd.DataFrame(), key_class = None, a_class = None):
    """TODO comment .it"""
    x, y, hue = f"bpp,psnr,{a_target}".split(",")
    x_, y_, hue_ = f"bpp,psnr [db],{a_target}".split(",")

    tmp_df = copy.deepcopy(a_df)

    flag_save: bool = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        flag_save = True
        pass
    ax.set_title(label=f"{x_.capitalize()} vs. {y_.capitalize()} wrt {hue_}")

    add_target_quality_range(
        a_df, tmp_jpeg_max_q, tmp_jpeg_min_q,
        conf_data, ax, a_x_key = 'bpp', a_y_key = 'psnr')
    
    add_some_data_examples_wrt_jpeg_wr(
        a_df, tmp_jpeg_max_q, tmp_jpeg_min_q, conf_data, ax, a_x_key = 'bpp', a_y_key = 'psnr', key_class = key_class, a_class = a_class
    )
    

    def change_targets(a_target):
        skip_targets = 'JPEG,SIREN'.split(",")
        if a_target in skip_targets: return a_target
        try:
            if '_' not in a_target:
                p1, p2, p3 = a_target.split(":")
                p1 = p1[3:]
                p3 = 'L' if p3.startswith("N") else 'C'
                return f"{p1}:{p2}:{p3}"
            else:
                try:
                    p1, p2, p3 = a_target.split(":")
                    p3, p4 = p3.split("_")
                    p1 = p1[3:]
                    p3 = 'L' if p3.startswith("N") else 'C'
                    return f"{p1}:{p2}:{p3}:{p4}"
                except:
                    return a_target
        except:
            print(a_target)
            sys.exit(0)
            pass
            
    
    tmp_df[f"{a_target}"] = list(map(change_targets, tmp_df[f"{a_target}"].values))

    _ = sns.scatterplot(data=tmp_df,
        x=f"{x}", y=f"{y}", ax=ax, hue=f"{hue}"
    )
    
    if teachers_bslns_df.shape[0] != 0:
        xval, yval = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
        cmprss = teachers_bslns_df["cmprss-class"].values[0]
        cmprss = cmprss.replace("Baseline", "TCH")
        # ax.scatter(x, y, label = f"{cmprss}-psnr={y:.2f},bpp={x:.2f})", s=100, marker='*', color='red')
        ax.scatter(xval, yval, label = f"{cmprss[0:3]}", s=100, marker='*', color='red')

        ymin, ymax = min(tmp_df["psnr"].values), max(tmp_df["psnr"].values)
        xmin, xmax = min(tmp_df["bpp"].values), max(tmp_df["bpp"].values)

        data_vlines=dict(x=xval, ymin=ymin,ymax=ymax, linestyle=":", alpha=0.7)
        # ax.vlines(**data_vlines)
        data_hlines=dict(y=yval,xmin=xmin,xmax=xmax, linestyle=":", alpha=0.7)
        # ax.hlines(**data_hlines)
        ax.text(xval, yval-3, f'TCH:\npsnr={yval:.2f}\nbpp={xval:.2f}', horizontalalignment='left',
            verticalalignment='top',)
        ax.legend()
        pass
    
    fig_name_path = None
    if flag_save:
        plt.grid(True)
        plt.legend()
        if out_dir_path:
            fig_name_path = os.path.join(
                out_dir_path,
                f"scatter_{x}_{y}_wrt_{a_target}.png"
            )
            plt.savefig(fig_name_path)
    else:
        ax.grid(True)
        ax.legend()
        pass
    # plt.show()
    return tmp_df, fig_name_path


def create_graphics(a_df, jpeg_df, siren_bsln_df, conf_data, args, out_dir_path, a_target, teachers_bslns_df = pd.DataFrame(), grid = (2, 2)):
    """TODO comment .it"""
    x, y, hue = f"bpp,psnr,{a_target}".split(",")
    x_, y_, hue_ = f"bpp,psnr [db],{a_target}".split(",")

    # Prepare data
    jpeg_tmp = jpeg_df[[x, y, "cmprss-class-2"]]
    jpeg_conf_dict = conf_data["exp_infos"]["filter_data"]["jpeg"]
    jpeg_tmp = filter_dataframe_by_conf(jpeg_tmp, conf_data_dict=jpeg_conf_dict, a_key = 'psnr')

    jpeg_filtered_df, tmp_jpeg_max_q, tmp_jpeg_min_q = \
        keep_target_qualities(jpeg_df, jpeg_tmp, jpeg_conf_dict)

    bs_tmp = siren_bsln_df[siren_bsln_df["bpp"] <= 12.0 ][[x, y, "cmprss-class-2"]]
    siren_bs_conf_dict = conf_data["exp_infos"]["filter_data"]["siren_bs"]
    bs_tmp = filter_dataframe_by_conf(bs_tmp, siren_bs_conf_dict, a_key = 'psnr')

    qatlrq_bs_conf_dict = conf_data["exp_infos"]["filter_data"]["qatrlq"]
    a_df = filter_qatlrq_conf(a_df, qatlrq_bs_conf_dict, tmp_jpeg_min_q)

    jpeg_tmp.columns = f"bpp,psnr,{a_target}".split(",")
    bs_tmp.columns = f"bpp,psnr,{a_target}".split(",")

    if grid:
        fig, axes = plt.subplots(grid[0], grid[1], figsize = (10, 10))
        if teachers_bslns_df.shape[0] != 0:
            x, y = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
            cmprss = teachers_bslns_df["cmprss-class"].values[0]
            cmprss = cmprss.replace("Baseline", "TCH")
            title = f"{cmprss}-(psnr={y:.2f},bpp={x:.2f})"
            fig.suptitle(f"{title} s.t. Graphics wrt {a_target}")
        else:
            fig.suptitle(f"Graphics wrt {a_target}")
        try:
            axes = list(itertools.chain.from_iterable(axes))
        except:
            pass
        pass

    x, y, hue = f"bpp,psnr,{a_target}".split(",")
    x_, y_, hue_ = f"bpp,psnr [db],{a_target}".split(",")
    tmp_df = pd.concat([a_df[[x, y, hue]], jpeg_tmp, bs_tmp], axis = 0, ignore_index = True)

    # Create scatter plot
    # ---------------------------------------------------------------------- #
    if grid:
        ax = axes[0]
    else:
        ax = None
    
    figs_list = []
    tmp_df, fig_name_path = \
        create_scatter_plot(
            tmp_df, conf_data, args,
            out_dir_path, teachers_bslns_df=teachers_bslns_df,
            a_target=a_target, ax = ax,
            tmp_jpeg_max_q=tmp_jpeg_max_q, tmp_jpeg_min_q=tmp_jpeg_min_q, key_class = a_target, a_class = 'SIREN')
    if not grid:
        figs_list.append(fig_name_path)

    # Create box plot (1)
    # ---------------------------------------------------------------------- #
    if grid:
        ax = axes[2]
    else:
        fig, ax  = plt.subplots(1, 1)
    x, y = f"{a_target},psnr".split(",")
    ax_box = sns.boxplot(x=x, y=y, data=tmp_df, ax = ax)
    ax_box.set_xticklabels(ax_box.get_xticklabels(),rotation=30)
    ax.set_title(f"boxplot of {y} wrt {x} ")
    if not grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"boxplot_{x}_{y}_wrt_{a_target}.png"
        )
        figs_list.append(fig_name_path)
        plt.savefig(fig_name_path)
    # plt.show()

    # Create box plot (2)
    # ---------------------------------------------------------------------- #
    if grid:
        ax = axes[3]
    else:
        fig, ax  = plt.subplots(1, 1)
    x, y = f"{a_target},bpp".split(",")
    ax_box = sns.boxplot(x=x, y=y, data=tmp_df, ax = ax)
    ax_box.set_xticklabels(ax_box.get_xticklabels(),rotation=30)
    ax.set_title(f"boxplot of {y} wrt {x} ")
    if not grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"boxplot_2_{x}_{y}_wrt_{a_target}.png"
        )
        figs_list.append(fig_name_path)
        plt.savefig(fig_name_path)
    # plt.show()

    # Create hist plot
    # ---------------------------------------------------------------------- #
    if grid:
        ax = axes[1]
    else:
        fig, ax  = plt.subplots(1, 1)
    x, y = f"{a_target},psnr".split(",")
    ax.set_title(f"histogram of {y} wrt {x} ")

    hue, x = f"{a_target},psnr".split(",")
    axes_hist = sns.histplot(data=tmp_df, x=x, kde=False, hue=hue, element="step", stat="density", ax=ax)
    
    if teachers_bslns_df.shape[0] != 0:
        x = teachers_bslns_df["psnr"].values[0]
        # x, y = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
        cmprss = teachers_bslns_df["cmprss-class"].values[0]
        cmprss = cmprss.replace("Baseline", "TCH")
        ymin, ymax = ax.get_ylim()
        axes_hist.vlines(x=x, ymin=ymin, ymax=ymax, label = f"{cmprss}", linestyle="-.")

        jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]
        jmax_psnr = tmp_jpeg_max_q["psnr"].values[0]
        axes_hist.vlines(x=jmin_psnr, ymin=ymin, ymax=ymax, linestyle=":", color = 'green')
        axes_hist.vlines(x=jmax_psnr, ymin=ymin, ymax=ymax, label = f"jpeg: min/max w.r.", color = 'green', linestyle=":")
        axes_hist.legend()
        pass
    if not grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"hist_{x}_{y}_wrt_{a_target}.png"
        )
        figs_list.append(fig_name_path)
        ax.legend()
        plt.savefig(fig_name_path)
        pass
    # plt.show()

    # Create hist plot (2)
    # ---------------------------------------------------------------------- #
    """
    if grid:
        ax = axes[3]
    else:
        fig, ax  = plt.subplots(1, 1)
    x, y = f"{a_target},psnr".split(",")
    ax.set_title(f"histogram of {y} wrt {x} ")

    hue, x, y = f"{a_target},bpp,psnr".split(",")
    axes_hist = sns.histplot(data=tmp_df, x=x, y=y, kde=False, hue=hue, element="step", stat="density", ax=ax)
    if teachers_bslns_df.shape[0] != 0:
        xval, yval = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
        cmprss = teachers_bslns_df["cmprss-class"].values[0]
        cmprss = cmprss.replace("Baseline", "TCH")
        # ax.scatter(x, y, label = f"{cmprss}-psnr={y:.2f},bpp={x:.2f})", s=100, marker='*', color='red')
        axes_hist.scatter( xval, yval, label = f"{cmprss}", s=100, marker='*', color='red')

        ymin, ymax = min(tmp_df["psnr"].values), max(tmp_df["psnr"].values)
        xmin, xmax = min(tmp_df["bpp"].values), max(tmp_df["bpp"].values)

        data_vlines=dict(x=xval, ymin=ymin,ymax=ymax, linestyle="-.", alpha=0.7)
        axes_hist.vlines(**data_vlines)
        data_hlines=dict(y=yval,xmin=xmin,xmax=xmax, linestyle="-.", alpha=0.7)
        axes_hist.hlines(**data_hlines)
        axes_hist.legend()
        pass
    if not grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"hist_2_{x}_{y}_wrt_{a_target}.png"
        )
        figs_list.append(fig_name_path)
        plt.savefig(fig_name_path)
        pass
    # plt.show()
    if grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"grid_picture.png"
        )
        figs_list.append(fig_name_path)
        plt.savefig(fig_name_path)
        pass
    """

    if grid:
        fig_name_path = os.path.join(
            out_dir_path,
            f"grid_picture.png"
        )
        figs_list.append(fig_name_path)
        plt.savefig(fig_name_path)
        pass

    # Creat Jointplot
    # ---------------------------------------------------------------------- #
    x, y, hue = f"bpp,psnr,{a_target}".split(",")
    fig, ax  = plt.subplots(1, 1)
    g = sns.JointGrid(data=tmp_df, x="bpp", y="psnr", hue=f"{a_target}",)
    """
    g = sns.jointplot(
        data=tmp_df,
        x="bpp", y="psnr", hue=f"{a_target}",
        kind="kde",
    )"""
    # g.plot(sns.scatterplot, sns.histplot)
    g.plot(sns.scatterplot, sns.histplot)
    fig_name_path = os.path.join(
        out_dir_path,
        f"joinplot_1_{x}_{y}_wrt_{a_target}.png"
    )
    if teachers_bslns_df.shape[0] != 0:
        xval, yval = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
        cmprss = teachers_bslns_df["cmprss-class"].values[0]
        cmprss = cmprss.replace("Baseline", "TCH")
        title = f"{cmprss}-(psnr={xval:.2f},bpp={yval:.2f})"
        ymin, ymax = 0, 60
        xmin, xmax = -2.5, 15.0
        # plt.vlines(x=xval, ymin=ymin, ymax=ymax, label = f"{cmprss}", linestyle="-.")
        # plt.hlines(y=yval, xmin=xmin, xmax=xmax, label = f"{cmprss}", linestyle="-.")
        g.fig.suptitle(f"{title} {x} vs {y} wrt {hue}")
        g.ax_joint.scatter( xval, yval, label = f"{cmprss}", s=200, marker='*', color='red')
        # g.ax_joint.vlines(x=xval, ymin=ymin, ymax=ymax, linestyle="-.", alpha=0.7)
        # g.ax_joint.hlines(y=yval, xmin=xmin, xmax=xmax, linestyle="-.", alpha=0.7)
        g.ax_joint.legend()
        pass
    else:
       g.fig.suptitle(f"Jointplot: {x} vs {y} wrt {hue}")
       pass
    add_target_quality_range(
        tmp_df, tmp_jpeg_max_q, tmp_jpeg_min_q,
        conf_data, ax = g.ax_joint, a_x_key = 'bpp', a_y_key = 'psnr')
    add_some_data_examples_wrt_jpeg_wr(
        tmp_df, tmp_jpeg_max_q, tmp_jpeg_min_q, conf_data, ax=g.ax_joint, a_x_key = 'bpp', a_y_key = 'psnr', key_class = a_target, a_class = 'SIREN'
    )
    g.ax_joint.legend()
    figs_list.append(fig_name_path)
    plt.savefig(fig_name_path)


    # Creat Jointplot (2)
    # ---------------------------------------------------------------------- #
    x, y, hue = f"bpp,psnr,{a_target}".split(",")
    fig, ax  = plt.subplots(1, 1)
    g = sns.JointGrid(data=tmp_df, x="bpp", y="psnr", hue=f"{a_target}",)
    """
    g = sns.jointplot(
        data=tmp_df,
        x="bpp", y="psnr", hue=f"{a_target}",
        kind="kde",
    )"""
    # g.plot(sns.scatterplot, sns.histplot)
    g.plot(sns.scatterplot, sns.kdeplot)
    fig_name_path = os.path.join(
        out_dir_path,
        f"joinplot_2_{x}_{y}_wrt_{a_target}.png"
    )
    if teachers_bslns_df.shape[0] != 0:
        xval, yval = teachers_bslns_df["bpp"].values[0], teachers_bslns_df["psnr"].values[0]
        cmprss = teachers_bslns_df["cmprss-class"].values[0]
        cmprss = cmprss.replace("Baseline", "TCH")
        title = f"{cmprss}-(psnr={xval:.2f},bpp={yval:.2f})"
        ymin, ymax = 0, 60
        xmin, xmax = -2.5, 15.0
        # plt.vlines(x=xval, ymin=ymin, ymax=ymax, label = f"{cmprss}", linestyle="-.")
        # plt.hlines(y=yval, xmin=xmin, xmax=xmax, label = f"{cmprss}", linestyle="-.")
        g.fig.suptitle(f"{title} {x} vs {y} wrt {hue}")
        g.ax_joint.scatter( xval, yval, label = f"{cmprss}", s=200, marker='*', color='red')
        g.ax_joint.vlines(x=xval, ymin=ymin, ymax=ymax, linestyle="-.", alpha=0.7)
        g.ax_joint.hlines(y=yval, xmin=xmin, xmax=xmax, linestyle="-.", alpha=0.7)
        g.ax_joint.legend()
        pass
    else:
       g.fig.suptitle(f"Jointplot: {x} vs {y} wrt {hue}")
       pass
    add_target_quality_range(
        tmp_df, tmp_jpeg_max_q, tmp_jpeg_min_q,
        conf_data, ax = g.ax_joint, a_x_key = 'bpp', a_y_key = 'psnr')
    g.ax_joint.legend()
    add_some_data_examples_wrt_jpeg_wr(
        tmp_df, tmp_jpeg_max_q, tmp_jpeg_min_q, conf_data, ax=g.ax_joint, a_x_key = 'bpp', a_y_key = 'psnr', key_class = a_target, a_class = 'SIREN'
    )
    g.ax_joint.legend()
    figs_list.append(fig_name_path)
    plt.savefig(fig_name_path)

    
    return figs_list


def create_baselines_images(conf_data, figures_list, args):
    """TODO comment .it"""
    jpeg_df, siren_bsln_df, baseline_df, _ = get_some_dfs()

    baselines_dir = conf_data["exp_infos"]["oud_dirs_info"]["baselines_dir"]
    out_dir_path = os.path.join(
        args.output_dir_path,
        f"{baselines_dir}")
    try: os.makedirs(out_dir_path)
    except: pass

    attributes = "psnr,ssim,CR".split(",")
    x = "bpp"
    fig = create_static_scatter_graphic(a_df = jpeg_df, attributes=attributes, x=x)
    image_path = os.path.join(out_dir_path, "complex_scatter_jpeg.png")
    plt.savefig(image_path)
    figures_list.append(image_path)

    attributes = "psnr,ssim,CR".split(",")
    x = "bpp"
    create_static_scatter_graphic(a_df = siren_bsln_df, attributes=attributes, x=x)
    image_path = os.path.join(out_dir_path, "complex_scatter_siren.png")
    plt.savefig(image_path)
    figures_list.append(image_path)

    root_dir_trained_baselines = '/home/franec94/Documents/thesys-siren/codebase/results/tests/datasets/tmp/baselines/pool_1610833309.4336185'
    dataset_name = "out_merged_1610833445.9098427.csv"
    a_dataset_path = os.path.join(root_dir_trained_baselines, dataset_name)
    dataset_bsln = get_selected_dataset(a_dataset_path=a_dataset_path)

    data=dict(
        root_dir = '/home/franec94/Documents/thesys-siren/codebase/results/datasets/tmp/baselines/pool_1610833309.4336185',
        dataset_name = "enanched_pool_1610833309.4336185.csv",
        shape=dataset_bsln.shape,
        # headers=list(dataset_bsln.columns)
    )
    meta_data  = dict(
        tabular_data=data.items(),
        tablefmt="github"
    )
    table = tabulate.tabulate(**meta_data)
    # print(table)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged_df = pd.concat([dataset_bsln, jpeg_df, siren_bsln_df], axis = 0, ignore_index = True)
        attributes = "psnr,ssim,CR".split(",")
        x = "bpp"
        fig = create_static_scatter_graphic(
            a_df = merged_df, attributes=attributes, x=x, hue="cmprss-class-2")
        image_path = os.path.join(out_dir_path, "complex_scatter_baseline_choice.png")
        plt.savefig(image_path)
        figures_list.append(image_path)
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged_df = pd.concat([dataset_bsln, jpeg_df, siren_bsln_df], axis = 0, ignore_index = True)
        attributes = "psnr".split(",")
        x = "bpp"
        fig = create_static_scatter_graphic(a_df = merged_df,
            attributes=attributes, x=x, hue="cmprss-class-2")
        image_path = os.path.join(out_dir_path, "scatter_baseline_choice.png")
        plt.savefig(image_path)
        figures_list.append(image_path)
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributes = "psnr".split(","); x = "bpp"
        merged_df = pd.concat([dataset_bsln, jpeg_df, siren_bsln_df[siren_bsln_df["bpp"] <= 14.0]], axis = 0, ignore_index = True)
        meta_data_boundaries = dict(
            jpeg_min_psnr=38.0,
            jpeg_max_psnr=max(jpeg_df["psnr"].values),
            image_bpp=8.0,
        )
        fig = create_static_scatter_graphic_with_boundaries(
            a_df = merged_df, attributes=attributes,
            x=x, hue="cmprss-class-2",
            meta_data_boundaries=meta_data_boundaries)
        image_path = os.path.join(out_dir_path, "scatter_baseline_choice_2.png")
        plt.savefig(image_path)
        figures_list.append(image_path)
        pass
    pass


def create_scatter_images(conf_data, figures_list, args):
    """TODO comment .it"""

    jpeg_df, siren_bsln_df, baseline_df, _ = get_some_dfs()

    camera, camera_crop, size_byte_crop, a_df, fetching_results = \
        merge_datasets_quant_data(conf_data)
    

    unique_pairs = set(map(lambda item: (item[0], item[1]), list(a_df[["n_hf", "n_hl"]].values)))

    # pprint(conf_data)
    # sys.exit(0)

    n_hf_t = conf_data["exp_infos"]["model_infos"]["n_hf"]
    n_hl_t = conf_data["exp_infos"]["model_infos"]["n_hl"]

    
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
    teachers_bslns_df = pd.concat(targtes_dfs_list, axis = 0, ignore_index = True)

    """
    out_dir_1 = conf_data["exp_infos"]["oud_dirs_info"]["out_dir_1"]
    out_dir_path = os.path.join(
        args.output_dir_path,
        f"{out_dir_1}")
    try: os.makedirs(out_dir_path)
    except: pass
    
    figs_list = \
        create_graphics(a_df, jpeg_df, siren_bsln_df, conf_data, args, out_dir_path, teachers_bslns_df=teachers_bslns_df, a_target='quant_techs')
    figures_list.extend(figs_list)
    
    out_dir_2 = conf_data["exp_infos"]["oud_dirs_info"]["out_dir_2"]
    out_dir_path = os.path.join(
        args.output_dir_path,
        f"{out_dir_2}")
    try: os.makedirs(out_dir_path)
    except: pass

    tmp_df, list_dfs = None, []
    for a_group, data in a_df.groupby(by = ["quant_techs", "lr"]):
        data["qt_lr"] = ['_'.join(list(map(str, a_group)))] * data.shape[0]
        list_dfs.append(data)
        pass
    tmp_df = pd.concat(list_dfs, ignore_index = True)
    figs_list = \
        create_graphics(tmp_df, jpeg_df, siren_bsln_df, conf_data, args, out_dir_path, teachers_bslns_df=teachers_bslns_df, a_target='qt_lr')
    figures_list.extend(figs_list)
    """

    out_dir_3 = conf_data["exp_infos"]["oud_dirs_info"]["out_dir_3"]
    out_dir_path = os.path.join(
        args.output_dir_path,
        f"{out_dir_3}")
    try: os.makedirs(out_dir_path)
    except: pass

    tmp_df, list_dfs = None, []
    for a_group, data in a_df.groupby(by = ["quant_techs", "lr"]):
        a_group = (a_group[0].split(":")[0][3:], a_group[1])
        pprint('_'.join(list(map(str, a_group))))
        # sys.exit(0)
        data["clss_lr"] = ['_'.join(list(map(str, a_group)))] * data.shape[0]
        list_dfs.append(data)
        pass
    tmp_df = pd.concat(list_dfs, ignore_index = True)
    figs_list = \
        create_graphics(tmp_df, jpeg_df, siren_bsln_df, conf_data, args, out_dir_path, teachers_bslns_df=teachers_bslns_df, a_target='clss_lr')
    figures_list.extend(figs_list)


    return a_df

# ---------------------------------------------- #
# Create Graphics About Input Image Stats
# ---------------------------------------------- #

def create_target_image_stats(im, im_cropped, im_cropped_size, args, conf_data, figures_list):
    """TODO comment .it"""
    try:
        target_images = conf_data["exp_infos"]["oud_dirs_info"]["target_images"]
    except: pass
    try:
        target_images = conf_data["target_images_dir"]
    except: pass
    out_dir_path = os.path.join(
        args.output_dir_path,
        f"{target_images}")
    try: os.makedirs(out_dir_path)
    except: pass
    image_path = os.path.join(out_dir_path, "original_image.png")
    im.save(image_path)
    figures_list.append(image_path)

    image_path = os.path.join(out_dir_path, "cropped_image.png")
    im_cropped.save(image_path)
    figures_list.append(image_path)

    fig = plt.figure()
    _ = plt.plot(im.histogram())
    plt.title(f"Histogram Camera Image({im.size[0]}x{im.size[1]})")
    plt.grid(True)
    image_path = os.path.join(out_dir_path, "full_image_hist.png")
    plt.savefig(image_path)
    figures_list.append(image_path)


    fig = plt.figure()
    _ = plt.plot(im_cropped.histogram())
    plt.grid(True)
    plt.title(f"Histogram Camera Image({im_cropped.size[0]}x{im_cropped.size[1]})")
    image_path = os.path.join(out_dir_path, "cropped_image_hist.png")
    plt.savefig(image_path)
    figures_list.append(image_path)

    fig, ax  = plt.subplots(1, 1, figsize = (15, 5))

    ax.plot(im.histogram(), label = f"Camera - Full ({im.size[0]}x{im.size[1]})")
    ax.plot(im_cropped.histogram(), label = f"Camera - Cropped ({im_cropped.size[0]}x{im_cropped.size[1]})")

    ax.set_xlabel("Pixel Intensity (Grayscale Image 0-255 levels)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Histogram Full Camera Image ({im.size[0]}x{im.size[1]}) vs. Cropped ({im_cropped.size[0]}x{im_cropped.size[1]})")

    plt.legend()
    plt.grid(True)
    image_path = os.path.join(out_dir_path, "mixing_image_hist.png")
    plt.savefig(image_path)
    figures_list.append(image_path)
    pass
