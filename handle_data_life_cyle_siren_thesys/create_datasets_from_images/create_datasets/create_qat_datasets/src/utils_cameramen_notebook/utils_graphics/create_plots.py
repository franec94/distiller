import PIL
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *
from src.data_loaders import dataset_loaders

from src.utils_cameramen_notebook.utils_graphics import utils_graphics as cameramen_ugraph
from src.utils_cameramen_notebook.utils_graphics import create_plots as cameramen_cplots

# ============================================================= #
# Utils for Adjust Dataframes
# ============================================================= #
def pick_some_baseline_rows(a_df:pd.DataFrame = pd.DataFrame(), pairs_hf_hl:list = []) -> pd.DataFrame:
    """Pick some baseline rows to be shown.\n
    Returns:
    --------
    `picked_bsln_rows_df` - pd.DataFrame
    """
    picked_bsln_rows_df = pd.DataFrame()
    empty_dataframe = pd.DataFrame()
    
    if a_df.shape[0] == 0: return empty_dataframe
    if pairs_hf_hl == []: return empty_dataframe

    try:
        picked_bsln_rows_list: list = []
        for ii, (n_hf, n_hl) in enumerate(pairs_hf_hl):
            pos = (a_df["n_hf"] == n_hf) & (a_df["n_hl"] == n_hl)

            picked_bsln_rows_list.append(a_df[pos].head(1))
            pass
        picked_bsln_rows_df = pd.concat(picked_bsln_rows_list, axis=0, ignore_index=True)
        pass
    except Exception as err:
        print(f"Error occuring when `pick_some_baseline_rows` function was called!")
        print(f"Returned Empty Dataframe, since following error occurs:")
        print(f"{str(err)}")
        return empty_dataframe

    picked_bsln_rows_df = adjust_baseline_df(a_df=picked_bsln_rows_df)
    return picked_bsln_rows_df


def get_baseline_row(a_df:pd.DataFrame = pd.DataFrame(), n_hf:int=64, n_hl:int=5) -> pd.DataFrame:
    """get baseline row.\n
    Returns:
    --------
    `a_baseline_row` - pd.DataFrane.\n
    """
    empty_dataframe = pd.DataFrame()
    if a_df.shape[0] == 0: return empty_dataframe
    try:
        def filter_required_data(item, n_hf=n_hf, n_hl=n_hl):
            return f"hf={str(n_hf)}" in item and f"hl={str(n_hl)}" in item
        vals = a_df["cmprss-class"].values
        pos = list(map(filter_required_data, vals))

        a_baseline_row = a_df[pos].head(1)

        a_baseline_row["prune_techs"] = ["BASELINE"] * a_baseline_row.shape[0]
        a_baseline_row["prune_rate"] = ["-"] * a_baseline_row.shape[0]
    except Exception as err:
        print(f"Error occuring when `get_baseline_row` function was called!")
        print(f"Returned Empty Dataframe, since following error occurs:")
        print(f"{str(err)}")
        return empty_dataframe
    return a_baseline_row


def get_pruned_row(a_pruned_df:pd.DataFrame = pd.DataFrame(), a_quanted_df:pd.DataFrame=pd.DataFrame()) -> pd.DataFrame:
    """get pruned target row.\n
    Returns:
    --------
    `a_pruned_row` - pd.DataFrane.\n
    """
    empty_dataframe = pd.DataFrame()
    if a_pruned_df.shape[0] == 0: return empty_dataframe
    if a_quanted_df.shape[0] == 0: return empty_dataframe

    try:
        a_date = a_quanted_df["init_from"].unique()[0]

        date_attr: str = "date"
        if date_attr not in a_pruned_df.columns:
            date_attr = "date_train"
            pass
        pos = a_pruned_df[f"{date_attr}"] == a_date
        a_pruned_row = a_pruned_df[pos].head(1)
        a_pruned_row["prune_techs"] = ["AGP"] * a_pruned_row.shape[0]
    except Exception as err:
        print(f"Error occuring when `get_pruned_row` function was called!")
        print(f"Returned Empty Dataframe, since following error occurs:")
        print(f"{str(err)}")
        return empty_dataframe
    return a_pruned_row


def adjust_baseline_df(a_df:pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """Adjust baseline siren dataframe for plot reasons.\n
    Returns:
    --------
    `adjuste_df` - pd.DataFrane.\n
    """
    empty_dataframe = pd.DataFrame()
    if a_df.shape[0] == 0: return a_df
    try:
        def create_hf_hl_attrs(item):
            _, n_hf, n_hl = item.split(":")

            n_hf = int(n_hf.split("=")[1])
            n_hl = int(n_hl.split("=")[1])
            
            return (n_hf, n_hl)
        pick_cols = "cmprss-class"
        dest_cols = ["n_hf", "n_hl"]
        # vals = a_df["cmprss-class"].values
        vals = a_df[pick_cols].values
        a_df[dest_cols] = list(map(create_hf_hl_attrs, vals))

        def create_deepness_attr(item):
            _, _, n_hl = item.split(":")
            n_hl = int(n_hl.split("=")[1])
            if n_hl <= 5: return "low"
            elif n_hl > 9: return "high"
            return "mid"
        pick_cols = "cmprss-class"
        dest_cols = "deepness"
        # vals = a_df["cmprss-class"].values
        # a_df["deepness"] = list(map(create_deepness_attr, vals))
        vals = a_df[pick_cols].values
        a_df[dest_cols] = list(map(create_deepness_attr, vals))

        def create_occurs_params_attr(item):
            bpp, psnr = item
            if psnr < 40: return "BL-U.P"
            elif bpp > 8: return "BL-O.P"
            return "BL-M."
        pick_cols = ["bpp", "psnr"]
        dest_cols = "occurs_params"
        vals = a_df[pick_cols].values
        # a_df["occurs_params"] = list(map(create_occurs_params_attr, vals))
        a_df[dest_cols] = list(map(create_occurs_params_attr, vals))
    except Exception as err:
        print(f"Error occuring when `adjust_baseline_df` function was called!")
        print(f"Returned Empty Dataframe, since following error occurs:")
        print(f"{str(err)}")
        return empty_dataframe
    return a_df


def get_best_quanted_rows(model_choices_df:pd.DataFrame, a_quant_df:pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """get best quanted rows.\n
    Returns:
    --------
    `best_quanted_rows_df` - pd.DataFrane.\n
    """
    empty_dataframe = pd.DataFrame()
    if a_quant_df.shape[0] == 0: return empty_dataframe
    try:
        best_quanted_rows_df: list = []
        if model_choices_df.shape[0] != 0:
            try:
                model_choices_df["quant_techs"] = "BL,AGP".split(",")
            except: pass
            best_quanted_rows_df: list = [model_choices_df]
            pass
        
        for gk, gdata in a_quant_df.groupby(by=["quant_techs_2", "nbits"]):
            tech, nbits = gk
            if nbits == 5: continue
            gdata_sorted = gdata.sort_values(by=["psnr"], ascending=False)
            best_quanted_rows_df.append(gdata_sorted.head(1))
            pass
        best_quanted_rows_df = pd.concat(best_quanted_rows_df, axis=0, ignore_index=True)

        best_quanted_rows_df["nbits"] = best_quanted_rows_df['nbits'].fillna(32)

        def adjust_quant_techs(item):
            res = list(filter(lambda char: char.upper() == char, item))
            return ''.join(res)
        vals = best_quanted_rows_df["quant_techs"].values
        best_quanted_rows_df["quant_techs"] = list(map(adjust_quant_techs, vals))
    except Exception as err:
        print(f"Error occuring when `get_best_quanted_rows` function was called!")
        print(f"Returned Empty Dataframe, since following error occurs:")
        print(f"{str(err)}")
        return empty_dataframe
    return best_quanted_rows_df

# ============================================================= #
# Utils for Adjust Dataframes
# ============================================================= #

def show_title(image_name:str = "cameramen", ax=None) -> None:
    """Show Title."""
    if ax is None: return
    ax.set_xlabel("Bpp", fontweight="bold")
    ax.set_ylabel("Pnsr [db]", fontweight="bold")
    ax.set_title(f"Psnr vs Bpp Scatter\nSiren-DNNs applied to {image_name.capitalize()}(256x256)", fontsize=20, fontweight="bold")

    plt.legend(title="Siren Case Study", loc="lower right")
    pass


def load_dataframes_to_be_plot(image_name = "cameramen") -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Load dataframe for plot reasons.\n
    Returns:
    --------
    `baseline_df` - pd.DataFrame.\n
    `jpeg_cameramen_df` - pd.DataFrame.\n
    `pruned_image_df` - pd.DataFrame.\n
    """
    if image_name == "cameramen":
        baseline_df = dataset_loaders.load_siren_baselines_dataset(dtype="dataframe")
        pos = baseline_df["bpp"] <= 12.0
        baseline_df = baseline_df[pos]
    else:
        baseline_df = pd.DataFrame()
    
    jpeg_cameramen_df = dataset_loaders.load_jpeg_dataset(dtype="dataframe", image_name=f"{image_name}")

    pruned_image_df = dataset_loaders.load_prunining_dataset(dtype="dataframe", image_name=f"{image_name}")
    pruned_image_df["image_name"] = [f"{image_name}"] * pruned_image_df.shape[0]
    return baseline_df, jpeg_cameramen_df, pruned_image_df


def show_image_jpeg_plain_siren_bpp_vs_psnr_scatter(
    a_quant_df:pd.DataFrame = pd.DataFrame(), image_details: dict = None, pairs_hf_hl: list = [], show_plot:bool=True) -> object:
    """Show Scatterplot for Bpp vs Psnr."""

    if not image_details:
        image_details = dict(
            image_name="cameramen",
            data_shown="jpeg_plain_siren",  
        )
        pass
    image_name = image_details["image_name"]
    data_shown = image_details["data_shown"]
    root_dir = image_details["root_dir"]
    try: os.makedirs(root_dir)
    except: pass
    fig_name = os.path.join(
        root_dir,
        f"{image_name}_{data_shown}.png"
    )

    baseline_df, jpeg_cameramen_df, pruned_image_df = \
        load_dataframes_to_be_plot(image_name=image_name)
    images_list=jpeg_cameramen_df["image_name"].unique()

    baseline_df = adjust_baseline_df(a_df=baseline_df)
    picked_bsln_rows_df = pick_some_baseline_rows(a_df=baseline_df, pairs_hf_hl=pairs_hf_hl)
    a_baseline_row = get_baseline_row(a_df=baseline_df)

    model_choices_df = pd.concat([a_baseline_row], axis = 0, ignore_index=True)
 
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # --------------------------------------------- #
    # Show Image
    # --------------------------------------------- #
    cameramen_ugraph.show_images_within_plot(
        pos_x=6.3, pos_y=30, pos_2_x=6.0, pos_2_y=40,
        delta_x=1.0, delta_2_x=0.5,
        delta_y_2=4.5, delta_y=4.8,
        images_list=images_list,
        ax=ax)

    # --------------------------------------------- #
    # Show DataFrames
    # --------------------------------------------- #
    cameramen_ugraph.show_baseline_df(a_df=baseline_df, ax=ax, thsd_list=[40, 8.0])
    cameramen_ugraph.show_merged_jpeg_data_points(a_df=jpeg_cameramen_df, ax=ax, gk="image_name")

    # --------------------------------------------- #
    # Show Fixed Data Points
    # --------------------------------------------- #
    images_list=jpeg_cameramen_df["image_name"].unique()

    qualities = [20, 45, 68, 85, 95]
    for image_name in jpeg_cameramen_df["image_name"].unique():
        cameramen_ugraph.add_jpeg_fixed_points(
            jpeg_df=jpeg_cameramen_df, image_name=image_name,
            qualities=qualities, ax=ax, horizontalalignment="right")
        pass

    cameramen_ugraph.show_baseline_fixed_points(
        a_df=baseline_df, ax=ax, pairs_hf_hl=pairs_hf_hl,
        horizontalalignment="left"
    )
    cameramen_ugraph.show_models_choices(a_df=model_choices_df, ax=ax)

    # --------------------------------------------- #
    # Show Table
    # --------------------------------------------- #
    if picked_bsln_rows_df.shape[0] != 0:
        pick_cols = ["n_hf", "n_hl", "occurs_params", "bpp", "psnr"]
        picked_bsln_rows_with_index = picked_bsln_rows_df.set_index(["deepness"])
        # pick_cols = ["n_hf", "n_hl", "bpp", "psnr"]
        # picked_bsln_rows_with_index = picked_bsln_rows_df.set_index(["deepness", "occurs_params"])
        dfi.export(picked_bsln_rows_with_index.round(2)[pick_cols], "mytable.png")
        mytable = PIL.Image.open("mytable.png")
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(mytable, cmap='gray',)
                , (6, 20)
                , frameon=False
            )
        )
        pass
    
    # --------------------------------------------- #
    # Show Vline Threshold
    # --------------------------------------------- #
    ymin, ymax = ax.get_ylim()
    ax.vlines(ymin=ymin, ymax=ymax, x = 8, linestyle="--", alpha=0.5)
    ax.text(y = ymax, x = 8, s="Image Info: 8-bpp", fontdict={'fontsize': 12, 'fontweight': 'bold'})

    # --------------------------------------------- #
    # Show Title
    # --------------------------------------------- #
    show_title(image_name=image_name, ax=ax)
    plt.savefig(fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig_name, fig, ax


def show_image_jpeg_pruned_siren_bpp_vs_psnr_scatter(
    a_quant_df:pd.DataFrame = pd.DataFrame(), image_details: dict = None, pairs_hf_hl: list = [], show_plot:bool=True) -> object:
    """Show Scatterplot for Bpp vs Psnr."""

    empty_dataframe = pd.DataFrame()

    if not image_details:
        image_details = dict(
            image_name="cameramen",
            data_shown="jpeg_plain_siren",  
        )
        pass
    image_name = image_details["image_name"]
    data_shown = image_details["data_shown"]
    root_dir = image_details["root_dir"]
    try: os.makedirs(root_dir)
    except: pass
    fig_name = os.path.join(
        root_dir,
        f"{image_name}_{data_shown}.png"
    )

    baseline_df, jpeg_cameramen_df, pruned_image_df = \
        load_dataframes_to_be_plot(image_name=image_name)
    pruned_image_df["image_name_2"] = [f"{image_name}-AGP"] * pruned_image_df.shape[0]

    images_list=jpeg_cameramen_df["image_name"].unique()

    baseline_df = adjust_baseline_df(a_df=baseline_df)
    # picked_bsln_rows_df = pick_some_baseline_rows(a_df=baseline_df, pairs_hf_hl=pairs_hf_hl)

    a_baseline_row = get_baseline_row(a_df=baseline_df)
    if a_baseline_row.shape[0] != 0:
        a_baseline_row["prune_techs"] = "BL"
        pass

    a_pruned_row = get_pruned_row(a_pruned_df=pruned_image_df, a_quanted_df=a_quant_df)
    if a_pruned_row.shape[0] != 0:
        a_pruned_row["prune_techs"] = "AGP"
        if "prune_rate" not in a_pruned_row.columns:
            a_pruned_row["prune_rate"] = float(a_pruned_row["cmprss-class"].values[0].split(":")[1]) / 100
            pass
        pass

    choices_list = [a_baseline_row, a_pruned_row]
    choices_list = list(filter(lambda item: item.shape[0] != 0, choices_list))
    if choices_list == []:
        model_choices_df = pd.DataFrame()
        best_pruned_rows_df = pd.DataFrame()
    else:
        model_choices_df = pd.concat(choices_list, axis = 0, ignore_index=True)
        best_pruned_rows_df = model_choices_df
        pass
 
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # --------------------------------------------- #
    # Show Image
    # --------------------------------------------- #
    cameramen_ugraph.show_images_within_plot(
        pos_x=6.3, pos_y=30, pos_2_x=6.0, pos_2_y=40,
        delta_x=1.0, delta_2_x=0.5,
        delta_y_2=4.5, delta_y=4.8,
        images_list=images_list,
        ax=ax)

    # --------------------------------------------- #
    # Show DataFrames
    # --------------------------------------------- #
    cameramen_ugraph.show_baseline_df(a_df=baseline_df, ax=ax, thsd_list=[40, 8.0])
    cameramen_ugraph.show_merged_jpeg_data_points(a_df=jpeg_cameramen_df, ax=ax, gk="image_name")
    cameramen_ugraph.show_merged_pruned_data_points(a_df=pruned_image_df, ax=ax, gk="image_name")


    # --------------------------------------------- #
    # Show Fixed Data Points
    # --------------------------------------------- #
    images_list=jpeg_cameramen_df["image_name"].unique()

    qualities = [20, 45, 68, 85, 95]
    for image_name in jpeg_cameramen_df["image_name"].unique():
        cameramen_ugraph.add_jpeg_fixed_points(
            jpeg_df=jpeg_cameramen_df, image_name=image_name,
            qualities=qualities, ax=ax, horizontalalignment="right")
        pass

    cameramen_ugraph.show_baseline_fixed_points(
        a_df=baseline_df, ax=ax, pairs_hf_hl=pairs_hf_hl,
        horizontalalignment="left"
    )

    def create_prune_rate_attr(item):
        prune_rate = float(item.split(":")[1]) / 100
        return prune_rate
    if "cmprss-class" in pruned_image_df.columns:
        vals = pruned_image_df["cmprss-class"].values
        pruned_image_df["prune_rate"] = list(map(create_prune_rate_attr, vals))
        pass
    cameramen_ugraph.add_fixed_prune_points(a_df=pruned_image_df, ax=ax, horizontalalignment="left")

    cameramen_ugraph.show_models_choices(a_df=model_choices_df, ax=ax)


    # --------------------------------------------- #
    # Show Table
    # --------------------------------------------- #
    if best_pruned_rows_df.shape[0] != 0:
        pick_cols = ["prune_techs", "bpp", "psnr"]
        dfi.export(best_pruned_rows_df.round(2)[pick_cols], "mytable.png")
        mytable = PIL.Image.open("mytable.png")
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(mytable, cmap='gray',)
                , (6, 20)
                , frameon=False
            )
        )
        pass

    # --------------------------------------------- #
    # Show Vline Threshold
    # --------------------------------------------- #
    ymin, ymax = ax.get_ylim()
    ax.vlines(ymin=ymin, ymax=ymax, x = 8, linestyle="--", alpha=0.5)
    ax.text(y = ymax, x = 8, s="Image Info: 8-bpp", fontdict={'fontsize': 12, 'fontweight': 'bold'})

    # --------------------------------------------- #
    # Show Title
    # --------------------------------------------- #
    show_title(image_name=image_name, ax=ax)
    plt.savefig(fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig_name, fig, ax


def show_image_jpeg_quanted_siren_bpp_vs_psnr_scatter(
    a_quant_df:pd.DataFrame = pd.DataFrame(), image_details: dict = None, pairs_hf_hl: list = [], show_plot:bool=True) -> object:
    """Show Scatterplot for Bpp vs Psnr."""

    if not image_details:
        image_details = dict(
            image_name="cameramen",
            data_shown="jpeg_plain_siren",  
        )
        pass
    image_name = image_details["image_name"]
    data_shown = image_details["data_shown"]
    root_dir = image_details["root_dir"]
    try: os.makedirs(root_dir)
    except: pass
    fig_name = os.path.join(
        root_dir,
        f"{image_name}_{data_shown}.png"
    )

    a_quant_df["quant_techs_2"] = [f"{image_name}-RLQ"] * a_quant_df.shape[0]

    baseline_df, jpeg_cameramen_df, pruned_image_df = \
        load_dataframes_to_be_plot(image_name=image_name)
    pruned_image_df["image_name_2"] = [f"{image_name}-AGP"] * pruned_image_df.shape[0]
    
    images_list=jpeg_cameramen_df["image_name"].unique()

    baseline_df = adjust_baseline_df(a_df=baseline_df)
    # picked_bsln_rows_df = pick_some_baseline_rows(a_df=baseline_df, pairs_hf_hl=pairs_hf_hl)

    a_baseline_row = get_baseline_row(a_df=baseline_df)
    if a_baseline_row.shape[0] != 0:
        a_baseline_row["prune_techs"] = "BL"
        pass

    a_pruned_row = get_pruned_row(a_pruned_df=pruned_image_df, a_quanted_df=a_quant_df)
    if a_pruned_row.shape[0] != 0:
        a_pruned_row["prune_techs"] = "AGP"
        if "prune_rate" not in a_pruned_row.columns:
            a_pruned_row["prune_rate"] = float(a_pruned_row["cmprss-class"].values[0].split(":")[1]) / 100
            pass
        pass

    choices_list = [a_baseline_row, a_pruned_row]
    choices_list = list(filter(lambda item: item.shape[0] != 0, choices_list))
    if choices_list == []:
        model_choices_df = pd.DataFrame()
    else:
        model_choices_df = pd.concat(choices_list, axis = 0, ignore_index=True)
        # model_choices_df["prune_techs"] = "BL,AGP".split(",")
        pass

    best_quanted_rows_df = get_best_quanted_rows(model_choices_df=model_choices_df, a_quant_df=a_quant_df)
 
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # --------------------------------------------- #
    # Show Image
    # --------------------------------------------- #
    cameramen_ugraph.show_images_within_plot(
        pos_x=6.3, pos_y=30, pos_2_x=6.0, pos_2_y=40,
        delta_x=1.0, delta_2_x=0.5,
        delta_y_2=4.5, delta_y=4.8,
        images_list=images_list,
        ax=ax)

    # --------------------------------------------- #
    # Show DataFrames
    # --------------------------------------------- #
    cameramen_ugraph.show_baseline_df(a_df=baseline_df, ax=ax, thsd_list=[40, 8.0])
    cameramen_ugraph.show_merged_jpeg_data_points(a_df=jpeg_cameramen_df, ax=ax, gk="image_name")
    cameramen_ugraph.show_merged_pruned_data_points(a_df=pruned_image_df, ax=ax, gk="image_name")
    cameramen_ugraph.show_quant_data_points(a_df=a_quant_df, ax=ax, gk=["image_name", "nbits"])

    # --------------------------------------------- #
    # Show Fixed Data Points
    # --------------------------------------------- #
    images_list=jpeg_cameramen_df["image_name"].unique()

    qualities = [20, 45, 68, 85, 95]
    for image_name in jpeg_cameramen_df["image_name"].unique():
        cameramen_ugraph.add_jpeg_fixed_points(
            jpeg_df=jpeg_cameramen_df, image_name=image_name,
            qualities=qualities, ax=ax, horizontalalignment="right")
        pass

    cameramen_ugraph.show_baseline_fixed_points(
        a_df=baseline_df, ax=ax, pairs_hf_hl=pairs_hf_hl,
        horizontalalignment="left"
    )

    if not "quant_pos_alignment" in image_details.keys():
        image_details["quant_pos_alignment"] = []
        pass
    if not "quant_on_a_line" in image_details.keys():
        image_details["quant_on_a_line"] = []
        pass
    cameramen_ugraph.add_fixed_qat_points(
        a_df=a_quant_df, ax=ax,
        pos_alignment=image_details["quant_pos_alignment"],
        on_a_line=image_details["quant_on_a_line"],
        horizontalalignment="center")

    def create_prune_rate_attr(item):
        prune_rate = float(item.split(":")[1]) / 100
        return prune_rate
    if "cmprss-class" in pruned_image_df.columns:
        vals = pruned_image_df["cmprss-class"].values
        pruned_image_df["prune_rate"] = list(map(create_prune_rate_attr, vals))
        pass
    cameramen_ugraph.add_fixed_prune_points(a_df=pruned_image_df, ax=ax, horizontalalignment="left")

    cameramen_ugraph.show_models_choices(a_df=model_choices_df, ax=ax)


    # --------------------------------------------- #
    # Show Table
    # --------------------------------------------- #
    x_tb, y_tb = 6, 20
    if "x_tb" in image_details.keys():
        x_tb = image_details["x_tb"]
    if "y_tb" in image_details.keys():
        y_tb = image_details["y_tb"]
        pass
    if best_quanted_rows_df.shape[0] != 0:
        pick_cols = ["nbits", "quant_techs", "bpp", "psnr"]
        dfi.export(best_quanted_rows_df.round(2)[pick_cols], "mytable.png")
        mytable = PIL.Image.open("mytable.png")
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(mytable, cmap='gray',)
                , (x_tb, y_tb)
                , frameon=False
            )
        )
    pass

    # --------------------------------------------- #
    # Show Vline Threshold
    # --------------------------------------------- #
    ymin, ymax = ax.get_ylim()
    ax.vlines(ymin=ymin, ymax=ymax, x = 8, linestyle="--", alpha=0.5)
    ax.text(y = ymax, x = 8, s="Image Info: 8-bpp", fontdict={'fontsize': 12, 'fontweight': 'bold'})

    # --------------------------------------------- #
    # Show Title
    # --------------------------------------------- #
    show_title(image_name=image_name, ax=ax)
    plt.savefig(fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig_name, fig, ax
