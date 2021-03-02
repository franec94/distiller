from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from src.libs.all_end_to_end_exp_analyses import *
from src.end_to_end_utils.create_out_dirs import add_target_quality_range, add_some_data_examples_wrt_jpeg_wr

from src.handle_raw_data.load_data_from_conf import get_reference_datasets
from src.graphics.create_graphics_for_report import * # show_pie_chart

import PyPDF2
import img2pdf
import fitz

def get_min_max_jpeg_df_data(conf_data: dict, jpeg_df: pd.DataFrame) -> pd.DataFrame:
    """Comment it.
    Returns:
    --------
    `pd.DataFrame` - required dataset.\n
    """
    conf_data_dict = conf_data
    _, tmp_jpeg_max_q, tmp_jpeg_min_q = \
        keep_target_qualities(jpeg_df, jpeg_df, conf_data_dict,)
    return tmp_jpeg_max_q, tmp_jpeg_min_q


def get_read_of_cols(a_df:pd.DataFrame, cols: list = ["Unnamed: 0"]) -> pd.DataFrame:
    for a_col in cols:
        if a_col in a_df.columns:
            a_df = a_df.drop([a_col], axis = 1)
    return a_df


def save_all_images_as_merged_pdf(figures_list: list, args):
    """TODO Comment it."""
    pdf_filename =  os.path.join(
        args.output_dir_path, "merged.pdf")
    
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


def check_graphics_2pie_2bars(
    a_df:pd.DataFrame, target_class:str="param_class", target_class_2:str="cmprss-class-2",
    x_attr: str = "bpp", y_attr_1: str = "deepness", y_attr_2: str = "param_class",
    show_fig: bool = False, save_fig: bool = False, file_name: str = "chart_complex.png",
    **kwargs
    ) -> object:
    
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize=(10, 10)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if axes.shape[0] > 1: axes = axes.flatten()

    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {y_attr_1.upper()} Classes", fontweight="bold"
        )
    )
    ax=axes[0]
    _ = show_pie_chart(
        a_df=a_df,
        group_attributes=f"{y_attr_1}".split(","),
        ax=ax,
        show_fig=False, **kwargs
    )
    ax=axes[1]
    _ = create_a_bar_plot_counts_ws_index(
        a_df = a_df,
        index=f"{y_attr_1}", y=f"{target_class_2}",
        figures_list=[], hue="",
        ax=ax,
        save_fig = False
    )

    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {y_attr_2} Classes", fontweight="bold"
        )
    )
    ax=axes[2]
    _ = show_pie_chart(
        a_df=a_df,
        group_attributes=f"{y_attr_2}".split(","),
        ax=ax,
        show_fig=False, **kwargs
    )

    ax=axes[3]
    _ = create_a_bar_plot_counts_ws_index(
        a_df = a_df,
        index=f"{y_attr_2}", y=f"{target_class_2}",
        figures_list=[], hue="",
        ax=ax,
        save_fig = False
    )
    
    if save_fig:
        try: os.makedirs(os.path.dirname(file_name))
        except Exception as err: pass
        # print(f"Saving file: {file_name}...")
        plt.savefig(file_name)
    if show_fig: plt.show()
    return fig, axes


def check_graphics_complex_charts_pbbv(
    a_df:pd.DataFrame, target_class:str="param_class", target_class_2:str="cmprss-class-2",
    x_attr: str = "bpp", y_attr: str = "psnr",
    show_fig: bool = False, save_fig: bool = False, file_name: str = "chart_complex.png",
    **kwargs
    ) -> object:
    """check_graphics_complex_charts.
    Returns:
    --------
    `fig` - figure object.\n
    `axes` - figure object's axes np.ndarray object.\n
    """
    
    # Figure setup:
    # -------------
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize=(10, 10)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if axes.shape[0] > 1: axes = axes.flatten()
    # a_df = copy.deepcopy(merged_df)

    # Pie chart setup:
    # ----------------
    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {target_class.upper()} Classes", fontweight="bold"
        )
    )
    ax=axes[0]
    _ = show_pie_chart(
        a_df=a_df,
        group_attributes=f"{target_class}".split(","),
        ax=ax,
        show_fig=True, **kwargs
    )
    
    # create_a_bar_plot_counts_ws_index chart setup:
    # ----------------------------------------------
    ax=axes[1]
    _ = create_a_bar_plot_counts_ws_index(
        a_df = a_df,
        index=f"{target_class}", y=f"{target_class_2}",
        figures_list=[], hue="",
        ax=ax,
        save_fig = False
    )

    # Violinplot chart setup:
    # --------------------
    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {target_class.upper()} Classes", fontweight="bold"
        )
    )
    ax=axes[3]
    x=target_class; y=y_attr; hue=target_class
    _ = create_a_violin_plot(
    # _ = create_a_box_plot(
        a_df=a_df, figures_list = [],
        x=f"{x}", y=f"{y}", hue=f"{hue}",
        ax=ax, save_fig=False, show_jitter=False,
        show_fig=True, # **kwargs
    )

    # Scatter chart setup:
    # --------------------
    ax=axes[2]
    hue = target_class
    _ = create_a_box_plot(
        a_df=a_df, figures_list = [],
        x=f"{x}", y=f"{y}", hue=f"{hue}",
        ax=ax, save_fig=False, show_jitter=False,
        show_fig=True, # **kwargs
    )

    
    
    if save_fig:
        try: os.makedirs(os.path.dirname(file_name))
        except Exception as err: pass
        # print(f"Saving file: {file_name}...")
        plt.savefig(file_name)
    if show_fig: plt.show()
    return fig, axes


def check_graphics_complex_charts(
    a_df:pd.DataFrame, target_class:str="param_class", target_class_2:str="cmprss-class-2",
    x_attr: str = "bpp", y_attr: str = "psnr",
    show_fig: bool = False, save_fig: bool = False, file_name: str = "chart_complex.png",
    **kwargs
    ) -> object:
    """check_graphics_complex_charts.
    Returns:
    --------
    `fig` - figure object.\n
    `axes` - figure object's axes np.ndarray object.\n
    """
    
    # Figure setup:
    # -------------
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize=(10, 10)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if axes.shape[0] > 1: axes = axes.flatten()
    # a_df = copy.deepcopy(merged_df)

    # Pie chart setup:
    # ----------------
    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {target_class.upper()} Classes", fontweight="bold"
        )
    )
    ax=axes[0]
    _ = show_pie_chart(
        a_df=a_df,
        group_attributes=f"{target_class}".split(","),
        ax=ax,
        show_fig=True, **kwargs
    )
    
    # create_a_bar_plot_counts_ws_index chart setup:
    # ----------------------------------------------
    ax=axes[1]
    _ = create_a_bar_plot_counts_ws_index(
        a_df = a_df,
        index=f"{target_class}", y=f"{target_class_2}",
        figures_list=[], hue="",
        ax=ax,
        save_fig = False
    )

    # Violinplot chart setup:
    # --------------------
    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] {target_class.upper()} Classes", fontweight="bold"
        )
    )
    ax=axes[2]
    x=target_class; y=y_attr; hue=target_class
    _ = create_a_violin_plot(
    # _ = create_a_box_plot(
        a_df=a_df, figures_list = [],
        x=f"{x}", y=f"{y}", hue=f"{hue}",
        ax=ax, save_fig=False, show_jitter=False,
        show_fig=True, # **kwargs
    )

    # Scatter chart setup:
    # --------------------
    ax=axes[3]
    hue = target_class
    _ = create_a_scatter_plot(
        a_df = a_df,
        x=f"{x_attr}", y=f"{y_attr}", ax=ax,
        figures_list=[], hue=f"{hue}",
        save_fig = False
    )

    jpeg_df:pd.DataFrame = load_jpeg_dataset(dtype="dataframe")
    jpeg_df_cp = copy.deepcopy(jpeg_df)
    jpeg_df_cp["param_class"] = ["JPEG"] * jpeg_df_cp.shape[0]

    conf_data = dict(
        min_psnr= 34.0,
        max_psnr= None,
        sampling_datapoints= 5,
        max_bpp= None,
        min_bpp= None,
        min_q= 68,
        max_q= 95,
    )
    
    tmp_jpeg_max_q, tmp_jpeg_min_q = get_min_max_jpeg_df_data(
        conf_data=conf_data, jpeg_df=jpeg_df_cp
    )
    add_target_quality_range(
        a_df=a_df, tmp_jpeg_max_q=tmp_jpeg_max_q, tmp_jpeg_min_q=tmp_jpeg_min_q, conf_data=conf_data,
        ax=ax
    )
    if target_class != "param_class_2":
        add_some_data_examples_wrt_jpeg_wr(
            a_df=a_df, tmp_jpeg_max_q=tmp_jpeg_max_q, tmp_jpeg_min_q=tmp_jpeg_min_q, conf_data=conf_data,
            ax=ax, key_class="cmprss-class-2", a_class="SIREN"
        )
    
    ax.legend()
    
    
    if save_fig:
        try: os.makedirs(os.path.dirname(file_name))
        except Exception as err: pass
        # print(f"Saving file: {file_name}...")
        plt.savefig(file_name)
    if show_fig: plt.show()
    return fig, axes


def create_table_via_groupby(a_df: pd.DataFrame = pd.DataFrame(),
    x:str=None, y:str=None) -> pd.DataFrame:
    """Create a dataframe table via groupby.
    Args:
    -----
    `a_df` - pd.DataFrame, default empty dataframe.\n
    `x` - main attribute for grouping.\n
    `y` - attribute to be restored as column after grouping.\n
    Returns:
    --------
    `pd.DataFrame` - either empy dataframe if input dataframe is also empty, or required dataframe.\n
    """
    if a_df.shape[0] == 0: return pd.DataFrame()
    
    gpby_cols = f"{x},{y}".split(",")
    dfs_groups = a_df.groupby(by = gpby_cols)
    tmp_df = dfs_groups.size().to_frame(name='counts')
    tmp_df = tmp_df.reset_index(f"{y}")

    a_dict = dict()
    nbits_set = set()
    for index, (nbits, counts) in zip(tmp_df.index, tmp_df.values):
        # print(index, nbits, counts)
        nbits_set.add(nbits)
        index_dict = a_dict.setdefault(index, dict())
        index_dict[str(nbits)] = int(counts)
        a_dict[index] = index_dict
        pass
    # pprint(a_dict)
    data = list(a_dict.values())
    index = list(a_dict.keys())
    # columns = "nbits".split(",")
    tmp_df = pd.DataFrame(data = data, index = index)
    return tmp_df


def creat_table_for_bar_plot_counts_ws_index(a_df: pd.DataFrame,
    x:str="quant_techs", y:str="nbits", as_heatmap: bool = False, cmap=None) -> object:
    """Create a dataframe table via groupby.
    Args:
    -----
    `a_df` - pd.DataFrame, default empty dataframe.\n
    `x` - main attribute for grouping. (Default: 'quant_techs')\n
    `y` - attribute to be restored as column after grouping. Default('nbits')\n
    Returns:
    --------
    `object` - 
      - either pd.DataFrame: empy dataframe if input dataframe is also empty, or required dataframe;\n
      - or sns.heatmap object;\n
      - or pd.DataFrame styled.\n
    """
    target_df = create_table_via_groupby(
        a_df=a_df, x=x, y=y
    )
    if as_heatmap:
        return sns.heatmap(target_df, annot=True)
    if cmap:
        # target_df.style.background_gradient(cmap='Blues')
        return target_df.style.background_gradient(cmap=f'{cmap}')
    return target_df


def fecth_all_report_datasets():
    jpeg_df:pd.DataFrame = load_jpeg_dataset(dtype="dataframe")
    jpeg_df = get_read_of_cols(jpeg_df)
    jpeg_df = adjust_jpeg_df(a_df=jpeg_df)

    sired_bsln_df:pd.DataFrame = load_siren_baselines_dataset(dtype="dataframe")
    sired_bsln_df = get_read_of_cols(sired_bsln_df)
    sired_bsln_df = adjust_baseline_df(a_df=sired_bsln_df)

    quant_df:pd.DataFrame = load_quant_dataset(conf_data=None, dtype="dataframe")
    quant_df = get_read_of_cols(quant_df)
    quant_df = adjust_quant_df(a_df=quant_df)
    return sired_bsln_df, jpeg_df, quant_df


def adjust_jpeg_df(a_df):

    def create_param_class_class(item):
        _, q = item.split(":")
        q = int(q)
        if q <= 50: return "low(q<=50)"
        elif q >= 80: return "high(q>=80)"
        return "mid(50<q<80)"
    # a_df["param_class"] = ["JPEG"] * a_df.shape[0]
    arr = a_df["cmprss-class"].values
    a_df["param_class_2"] = list(map(create_param_class_class,arr))
    a_df["param_class"] = ["JPEG"] * a_df.shape[0]
    a_df["dp_pclss"] = ["JPEG"] * a_df.shape[0]
    a_df["deepness"] = ["JPEG"] * a_df.shape[0]
    return a_df


def adjust_quant_df(a_df):
    a_df["deepness"] = a_df["quant_techs"].values
    a_df["dp_pclss"] = a_df["quant_techs"].values
    a_df["cmprss-class-2"] = a_df["quant_techs"].values
    a_df["param_class"] = a_df["quant_techs"].values

    pos = a_df["psnr"] >= 35
    return a_df[pos]


def adjust_baseline_df(a_df):

    def create_deepness_class(item):
        _, n_hf, n_hl = item.split(":")
        n_hl = int(n_hl.split("=")[1])
        if n_hl <= 5: return "S-tiny(<=5)"
        elif n_hl > 9: return "S-deep(>=9)"
        return "S-mid(5~9)"
    
    cmprss_class_values = a_df["cmprss-class"].values
    a_df["deepness"] = list(map(create_deepness_class, cmprss_class_values))

    # Add 'param_class' attribute
    pos = a_df["bpp"] >= 8.
    over_size = a_df[pos].sort_values(by=["psnr"])["size(byte)"].values[0]
    pos = a_df["psnr"] <= 38.
    under_size = a_df[pos].sort_values(
        by=["psnr"], ascending=False)["size(byte)"].values[0]
    def create_params_no_class(item, ref_bpp=8., ref_psnr = 38.,
        over_size=over_size, under_size=under_size):
        bpp, psnr = item
        if bpp > ref_bpp : return f"S-over(size>={over_size}B)"
        if psnr < ref_psnr : return f"S-under(size<={under_size}B)"
        return f"S-mid([{under_size}~{over_size}]B)"
    arr_values = a_df[["bpp", "psnr"]].values
    a_df["param_class"] = list(map(create_params_no_class, arr_values))


    def create_params_no_mixed_deepness_class(item):
        deepness, param_class = item
        
        deepness_classes = "tiny,mid,deep".split(",")
        for a_clss_d in deepness_classes:
            if a_clss_d in deepness: break
            pass
        param_classes = "under,mid,over".split(",")
        for a_clss_p in param_classes:
            if a_clss_p in param_class: break
            pass
        dp_pclss = f"d:{a_clss_d}/p:{a_clss_p}"
        return dp_pclss

    arr_values = a_df[["deepness", "param_class"]].values
    a_df["dp_pclss"] = list(map(create_params_no_mixed_deepness_class, arr_values))
    return a_df


def get_initial_table_jpeg(a_df):
    pick_cols = "size(byte),footprint(%),cmprss-class,psnr,bpp,CR".split(",")

    cmprss_classes = "JPEG:20,JPEG:95,JPEG:70".split(",")
    pos = list(map(lambda item: item in cmprss_classes, a_df["cmprss-class"].values))
    a_df_cp = copy.deepcopy(a_df[pos][pick_cols])
    a_df_cp.index = "small,medium,large".split(",")
    
    # return a_df_cp.to_latex(index=True)
    return a_df_cp
