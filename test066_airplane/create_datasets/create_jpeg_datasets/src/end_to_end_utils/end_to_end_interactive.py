from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *
from src.quant_models_utils.create_comparing_qat_tables import create_comparing_results_table

from src.quant_models_utils.quant_tech_utils import get_list_of_quanted_datasets, get_selected_dataset

from ipywidgets import Button, HBox, VBox

UNWANTED_COLS = "Unnamed: 0,Unnamed: 0.1".split(",")
OLD_COLUMNS = "date_train,init_from,size_byte,footprint".split(",")
NEW_COLUMNS = "date,init-from,size(byte),footprint(%)".split(",")
TARGTE_CLASSES = "cmprss-class,cmprss-class-2,cmprss-class-3,prune_rate_intervals".split(",")

ROOT_DIR = "/home/franec94/Documents/thesys-siren/codebase/notebooks/analyze_quant_data/uniform_csv_files/tmp_quantized_datasets"

# ==================================================================================================== #
# Function: `pick_tmp_dataset_wrapper`
# ==================================================================================================== #

def get_rid_of_unwanted_columns(a_df: pd.DataFrame) -> pd.DataFrame:
    """TODO comment .it"""
    for a_col in UNWANTED_COLS:
        if f"{a_col}" in a_df.columns:
            a_df = a_df.drop([f"{a_col}"], axis = 1)
            pass
        pass
    return a_df


def rename_columns_and_get_rid_of_old_ones(a_df: pd.DataFrame, old_columns:list, new_columns:list) -> pd.DataFrame:
    """TODO comment .it"""
    for a_old_col, a_new_col in list(zip(old_columns, new_columns)):
        if a_old_col not in a_df.columns:
            continue
        a_df[f"{a_new_col}"] = a_df[f"{a_old_col}"]
        a_df = a_df.drop([f"{a_old_col}"], axis = 1)
        pass
    return a_df

                                           
def pick_tmp_dataset_wrapper(dataset_path_wg):
    """TODO comment .it"""

    all_quanted_csv_list, quanted_csv_dict = get_list_of_quanted_datasets()
    @interact
    def pick_tmp_dataset(datasets_list=all_quanted_csv_list):

        if datasets_list.lower() == 'All'.lower():
            a_dataset_path = ROOT_DIR
        else:
            a_dataset_path = quanted_csv_dict[datasets_list]
        
        dataset_path_wg.value = a_dataset_path
        tmp_df = get_selected_dataset(a_dataset_path)
        
        data = dict(
            df_root_dir=os.path.dirname(a_dataset_path),
            df_name=os.path.basename(a_dataset_path),
            df_shape=tmp_df.shape
        )
        meta_t = dict(
            tabular_data=data.items()
        )
        table = tabulate.tabulate(**meta_t)
        print(table)
        pass
    pass

# ==================================================================================================== #
# Function: `pick_local_file_system_tmp_dataset`
# ==================================================================================================== #

def pick_local_file_system_tmp_dataset(root_dir_tmp_datasets, QAT_DATASET_PATH):
    """TODO comment .it"""
    datasets_list = list(filter(lambda item: item.endswith(".csv"),os.listdir(root_dir_tmp_datasets)))
    QAT_DATASET_PATH = os.path.join(root_dir_tmp_datasets, datasets_list[0])
    @interact
    def pick_tmp_dataset(pick_dataset=datasets_list):
        QAT_DATASET_PATH = os.path.join(root_dir_tmp_datasets, pick_dataset)
        print(f"Dataset picked: {QAT_DATASET_PATH}")
        a_df = pd.read_csv(QAT_DATASET_PATH)
        if "Unnamed: 0" in a_df.columns:
            a_df = a_df.drop(["Unnamed: 0"], axis = 1)
            pass
        tabular_data = [
            ["Path", root_dir_tmp_datasets],
            ["Dataset Name", pick_dataset],
            ["Dataset Shape", a_df.shape],
            ["Columns", str(list(a_df.columns))],
            ["Quant Techs", str(list(set(a_df["cmprss-class"])))],
            ["Quant Techs-2", str(list(set(a_df["cmprss-class-2"])))],
        ]
        metadata_table = dict(
            tabular_data=tabular_data
        )
        table = tabulate.tabulate(**metadata_table)
        print(table)
        pass
    pass

# ==================================================================================================== #
# Function: `show_hist_or_pie_quant_tech`
# ==================================================================================================== #

def show_hist_or_pie_quant_tech(a_df: pd.DataFrame):
    """TODO comment .it"""
    # qat_df["cmprss-class-2"].iplot(kind="histogram", bins=20, theme="white", title="cmprss-class-2", xTitle='cmprss-class-2', yTitle='Count')
    templates_list = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    @interact
    def scatter_plot(hist_or_pie_chart=True,
        attribute=list(a_df.columns), theme=list(cf.themes.THEMES.keys()),
        colorscale=list(cf.colors._scales_names.keys()),
        templates_pie=templates_list):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if hist_or_pie_chart:
                a_df[f"{attribute}"].iplot(
                    kind="histogram", bins=20,
                    title=f"{attribute}", xTitle=f"{attribute}", yTitle='Count',
                    theme=theme, colorscale=colorscale)
                pass
            else:
                occr_dict = dict(collections.Counter(a_df[f"{attribute}"].values))
                keys, values = list(occr_dict.keys()), list(occr_dict.values())
                data = np.array(values)[:, np.newaxis]
                # .iplot(kind="pie", theme="white", y = "Occr")
                tmp_df = pd.DataFrame(data = data, index = keys, columns = ["Occr"])
                # tmp_df.plot.pie(y='Occr', figsize=(5, 5))

                fig = px.pie(tmp_df, values='Occr', names=tmp_df.index, template = templates_pie)
                fig.show()
                pass
        pass
    pass


def show_hist_or_pie_quant_tech_2(a_df:pd.DataFrame):
    """TODO comment .it"""
    # qat_df["cmprss-class-3"].iplot(kind="histogram", bins=20, theme="white", title="cmprss-class-2", xTitle='cmprss-class-2', yTitle='Count')
    templates_list = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    @interact
    def scatter_plot(hist_or_pie_chart=True, theme=list(cf.themes.THEMES.keys()), 
                     colorscale=list(cf.colors._scales_names.keys()),
                     templates_pie=templates_list):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if hist_or_pie_chart:
                a_df["cmprss-class-3"].iplot(
                    kind="histogram", bins=20,
                    title="cmprss-class-3", xTitle='cmprss-class-3', yTitle='Count',
                    theme=theme, colorscale=colorscale)
                pass
            else:
                templates_list = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
                occr_dict = dict(collections.Counter(a_df["cmprss-class-3"].values))
                keys, values = list(occr_dict.keys()), list(occr_dict.values())
                data = np.array(values)[:, np.newaxis]
                # .iplot(kind="pie", theme="white", y = "Occr")
                tmp_df = pd.DataFrame(data = data, index = keys, columns = ["Occr"])
                fig = px.pie(tmp_df, values='Occr', names=tmp_df.index, template = templates_pie)
                fig.show()
                pass
        pass
    pass

# ==================================================================================================== #
# Function: `show_scatter_plot`
# ==================================================================================================== #

def show_scatter_plot(a_df:pd.DataFrame, jpeg_df:pd.DataFrame = None, optional_dfs_dict: dict = None, target_class:str = 'cmprss-class-2'):
    """TODO comment .it"""
    targte_class = set(TARGTE_CLASSES[1:]).intersection(set(list(a_df.columns)))
    @interact
    def scatter_plot(
        flag_add_jpeg_data=False, flag_add_siren_data=False,
        x=list(a_df.select_dtypes('number').columns),
        y=list(a_df.select_dtypes('number').columns)[:],
        target_class=list(targte_class),
        theme=list(cf.themes.THEMES.keys()), 
        colorscale=list(cf.colors._scales_names.keys())):
        
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            title = f'{y.capitalize()} vs {x.capitalize()}'
            tmp_df = a_df[(a_df["bpp"] <= 12.0)]
            if flag_add_jpeg_data:
                title += f' + Jpge'
                if optional_dfs_dict:
                    tmp_df = pd.concat([optional_dfs_dict["jpeg_df"], tmp_df], axis = 0)
                else:
                    tmp_df = pd.concat([jpeg_df, tmp_df], axis = 0)
            if flag_add_siren_data:
                if optional_dfs_dict:
                    title += f' + Siren'
                    tmp_df = pd.concat([optional_dfs_dict["siren_df"], \
                        tmp_df], axis = 0)
            title += f' w.r.t {target_class}'
            tmp_df.iplot(kind='scatter', x=x, y=y, mode='markers', 
                xTitle=x.capitalize(), yTitle=y.capitalize(),
                text='cmprss-class', title=title,
                theme=theme, colorscale=colorscale, categories=f"{target_class}")
            pass
        pass
    pass


def show_scatter_plot_2(a_df, jpeg_df, optional_dfs_dict = None, target_class = 'cmprss-class-3'):
    @interact
    def scatter_plot(
        flag_add_jpeg_data=False, flag_add_siren_data=False,
        x=list(a_df.select_dtypes('number').columns),
        y=list(a_df.select_dtypes('number').columns)[:],
        theme=list(cf.themes.THEMES.keys()), 
        colorscale=list(cf.colors._scales_names.keys())):
        
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            title = f'{y.capitalize()} vs {x.capitalize()}'
            tmp_df = a_df[(a_df["bpp"] <= 12.0)]
            if flag_add_jpeg_data:
                title += f' + Jpge'
                if optional_dfs_dict:
                    tmp_df = pd.concat([optional_dfs_dict["jpeg_df"], tmp_df], axis = 0)
                else:
                    tmp_df = pd.concat([jpeg_df, tmp_df], axis = 0)
            if flag_add_siren_data:
                if optional_dfs_dict:
                    title += f' + Siren'
                    tmp_df = pd.concat([optional_dfs_dict["siren_df"], \
                        tmp_df], axis = 0)
            title += f' w.r.t {target_class}'
            tmp_df.iplot(kind='scatter', x=x, y=y, mode='markers', 
                xTitle=x.capitalize(), yTitle=y.capitalize(),
                text='cmprss-class', title=title,
                theme=theme, colorscale=colorscale, categories=f"{target_class}")
            pass
        pass
    pass

# ==================================================================================================== #
# Function: `show_box_plot`
# ==================================================================================================== #

def show_box_plot(a_df:pd.DataFrame, jpeg_df:pd.DataFrame, optional_dfs_dict:dict = None, target_class:str = 'cmprss-class-2'):
    """TODO comment .it"""
    # qat_df[['cmprss-class-3', 'psnr']].pivot(columns='cmprss-class-3', values='psnr').iplot(kind='box')
    targte_class = set(TARGTE_CLASSES).intersection(set(list(a_df.columns)))
    @interact
    def scatter_plot(flag_add_jpeg_data=False, flag_add_siren_data=False,
                     attribute=list(a_df.select_dtypes('number').columns),
                     target_class=list(targte_class),
                     theme=list(cf.themes.THEMES.keys()), colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            title = f'{attribute.capitalize()}'
            tmp_df = a_df[(a_df["bpp"] <= 12.0)]
            if flag_add_jpeg_data:
                title += f' + Jpge'
                if optional_dfs_dict:
                    tmp_df = pd.concat([optional_dfs_dict["jpeg_df"], tmp_df], axis = 0)
                else:
                    tmp_df = pd.concat([jpeg_df, tmp_df], axis = 0)
            if flag_add_siren_data:
                if optional_dfs_dict:
                    title += f' + Siren'
                    tmp_df = pd.concat([optional_dfs_dict["siren_df"], \
                        tmp_df], axis = 0)
            title +=  f' w.r.t {target_class}'
            tmp_df[[f'{target_class}', f'{attribute}']]\
                .pivot(columns=f'{target_class}', values=f'{attribute}')\
                .iplot(kind='box', theme=theme, colorscale=colorscale, title = title)
            pass
        pass
    pass


def show_box_plot_2(a_df, jpeg_df, optional_dfs_dict = None, target_class = 'cmprss-class-3'):
    # qat_df[['cmprss-class-3', 'psnr']].pivot(columns='cmprss-class-3', values='psnr').iplot(kind='box')
    @interact
    def scatter_plot(flag_add_jpeg_data=False, flag_add_siren_data=False,
                     attribute=list(a_df.select_dtypes('number').columns),
                     theme=list(cf.themes.THEMES.keys()), colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            title = f'{attribute.capitalize()}'
            tmp_df = a_df[(a_df["bpp"] <= 12.0)]
            if flag_add_jpeg_data:
                title += f' + Jpge'
                if optional_dfs_dict:
                    tmp_df = pd.concat([optional_dfs_dict["jpeg_df"], tmp_df], axis = 0)
                else:
                    tmp_df = pd.concat([jpeg_df, tmp_df], axis = 0)
            if flag_add_siren_data:
                if optional_dfs_dict:
                    title += f' + Siren'
                    tmp_df = pd.concat([optional_dfs_dict["siren_df"], \
                        tmp_df], axis = 0)
            title +=  f' w.r.t {target_class}'
            tmp_df[[f'{target_class}', f'{attribute}']]\
                .pivot(columns=f'{target_class}', values=f'{attribute}')\
                .iplot(kind='box', theme=theme, colorscale=colorscale, title = title)
            pass
        pass
    pass


def show_box_plot_static_wrapper(a_df: pd.DataFrame, jpeg_df:pd.DataFrame = None, optional_dfs_dict:dict = None, target_class:str = 'cmprss-class-3'):
    """TODO Comment it."""
    # qat_df[['cmprss-class-3', 'psnr']].pivot(columns='cmprss-class-3', values='psnr').iplot(kind='box')
    targte_class = set(TARGTE_CLASSES[1:]).intersection(set(list(a_df.columns)))
    @interact
    def scatter_plot(flag_add_jpeg_data=False, flag_add_siren_data=False,
                     attribute_1=list(a_df.select_dtypes('number').columns),
                     attribute_2=list(a_df.select_dtypes('number').columns),
                     target_class=list(targte_class),
                     theme=list(cf.themes.THEMES.keys()), colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            title = f'{attribute_1.capitalize()} vs. {attribute_2.capitalize()}'
            tmp_df = a_df[(a_df["bpp"] <= 12.0)]
            file_name = "static_box_plot"
            if flag_add_jpeg_data:
                title += f' + Jpge'
                file_name += "_jpeg"
                if optional_dfs_dict:
                    tmp_df = pd.concat([optional_dfs_dict["jpeg_df"], tmp_df], axis = 0)
                else:
                    tmp_df = pd.concat([jpeg_df, tmp_df], axis = 0)
            if flag_add_siren_data:
                if optional_dfs_dict:
                    title += f' + Siren'
                    file_name += "_siren"
                    tmp_df = pd.concat([optional_dfs_dict["siren_df"], \
                        tmp_df], axis = 0)
            title +=  f' w.r.t {target_class}'
            fig, axes  = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(title)

            ax = axes[0]
            x, y = f"{target_class}", f"{attribute_1}"
            ax.set_title(f"{attribute_1.capitalize()}")
            _ = sns.boxplot(x=f"{x}", y=f"{y}", data=tmp_df, ax = ax,)

            x, y = f"{target_class}", f"{attribute_2}"
            ax = axes[1]
            ax.set_title(f"{attribute_2.capitalize()}")
            _ = sns.boxplot(x=f"{x}", y=f"{y}", data=tmp_df, ax = ax)

            if not os.path.exists("./resources") or not os.path.isdir("./resources"):
                try: os.makedirs("./resources")
                except: pass
                pass
            plt.savefig(f"./resources/{file_name}.png")
            plt.show()
            pass
        pass
    pass

# ==================================================================================================== #
# Function: `create_resulting_table_for_comparison_reasons`
# ==================================================================================================== #

def create_resulting_table_for_comparison_reasons(table_df:pd.DataFrame) -> None:
    """TODO comment .it"""
    metadata_table = dict(
        table_df=table_df,
        show_columns="date,init-from,cmprss-class,size(byte),footprint(%),psnr,bpp,CR,ssim,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(","),
        delta_columns="footprint(%),psnr,bpp,CR,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(","),
        formatter_dict={'size(byte)': "{:.0f}",'ssim':'{:.3e}'},
        extend_formatter_keys="footprint(%),psnr,bpp,CR,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(",")
    )
    cmap_list = "'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'"
    cmap_list = [item.strip(" ").strip("''") for item in cmap_list.split(",")]
    @interact
    def table_comparing(hide_index=True, hide_date_init_from=False, enable_cmap=True, show_deltas=True, cmap=cmap_list):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            metadata_table['hide_index'] = hide_index
            metadata_table['cmap'] = cmap
            metadata_table['enable_cmap'] = enable_cmap
            metadata_table["show_deltas"] = show_deltas
            metadata_table["hide_date_init_from"] = hide_date_init_from
            resulting_table_df = create_comparing_results_table(**metadata_table)
            display(resulting_table_df)
        pass
    pass


def get_best_siren_pruned_rows_by_target_as_df(cmprss_df:pd.DataFrame, a_class:str, best_attr:str = "psnr") -> pd.DataFrame:
    """TODO comment .it"""
    list_df = []
    for gp_name, gp_data in cmprss_df.groupby(by = [f'{a_class}']):
        # print(gp_name)
        psnr_max = max(gp_data[f"{best_attr}"].values)
        psnr_min = min(gp_data[f"{best_attr}"].values)
        first_row_max_psnr = gp_data[gp_data[f"{best_attr}"] == psnr_max].iloc[0,:]
        first_row_min_psnr = gp_data[gp_data[f"{best_attr}"] == psnr_min].iloc[0,:]
        # list_df.append(first_row_max_psnr)
        mid_row_min_psnr = gp_data.iloc[gp_data.shape[0]//2,:]
        list_df.extend([first_row_max_psnr, mid_row_min_psnr, first_row_min_psnr])
        pass
    siren_prune_best_df = pd.DataFrame(list_df)
    return siren_prune_best_df


def get_best_jpeg_lower_equal_than_pruned_models(jpeg_df:pd.DataFrame, siren_prune_best_df:pd.DataFrame, best_attr:str = "psnr") -> pd.DataFrame:
    """TODO comment .it"""
    tmp_jpeg_df = jpeg_df[jpeg_df[f"{best_attr}"] <= max(siren_prune_best_df[f"{best_attr}"].values)]
    best_jpeg_equal_lower_prune_data = tmp_jpeg_df.sort_values(by = [f"{best_attr}"], ascending = False).iloc[0,:]

    columns = best_jpeg_equal_lower_prune_data.index
    data = [best_jpeg_equal_lower_prune_data.values]
    best_jpeg_equal_lower_prune_data = pd.DataFrame(data = data, columns = columns)
    
    return best_jpeg_equal_lower_prune_data


def create_resulting_table_for_comparison_reasons_2(
    baseline_data_df:pd.DataFrame, cmprss_df:pd.DataFrame,
    jpeg_df:pd.DataFrame = None, agp_df:pd.DataFrame = None) -> None:
    """TODO comment .it"""
    best_nth_jpge_df = jpeg_df.tail(3)

    cmap_list = "'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'"
    cmap_list = [item.strip(" ").strip("''") for item in cmap_list.split(",")]

    targte_class = set(TARGTE_CLASSES).intersection(set(list(cmprss_df.columns)))
    @interact
    def table_comparing(
        target_class=list(targte_class),
        hide_index=True, hide_date_init_from=False,
        enable_cmap=True, show_deltas=True, cmap=cmap_list):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            siren_prune_best_df = get_best_siren_pruned_rows_by_target_as_df(
                cmprss_df = cmprss_df,
                a_class = target_class, best_attr = "psnr")
            best_jpeg_equal_lower_prune_data = \
                get_best_jpeg_lower_equal_than_pruned_models(jpeg_df, siren_prune_best_df, best_attr = "psnr")
            if type(agp_df) == pd.core.frame.DataFrame:
                best_agp_equal_lower_prune_data = \
                    get_best_jpeg_lower_equal_than_pruned_models(agp_df, siren_prune_best_df, best_attr = "psnr")
                
                list_dfs = [baseline_data_df.sort_values(by=["psnr"]).head(3),
                    siren_prune_best_df,
                    best_agp_equal_lower_prune_data,
                    best_nth_jpge_df.sort_values(by=["psnr"], ascending=False),
                    best_jpeg_equal_lower_prune_data,
                ]
            else:
                list_dfs = [baseline_data_df.sort_values(by=["psnr"]).head(3),
                    siren_prune_best_df,
                    best_nth_jpge_df.sort_values(by=["psnr"], ascending=False),
                    best_jpeg_equal_lower_prune_data,
                ]
                pass
            table_df = pd.concat(list_dfs, axis = 0, ignore_index = True)

            show_columns = f"date,init-from,cmprss-class,size(byte),footprint(%),psnr,bpp,CR,ssim,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(",")
            show_columns = show_columns \
                if f"{target_class}" in show_columns \
                else f"date,init-from,{target_class},cmprss-class,size(byte),footprint(%),psnr,bpp,CR,ssim,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(",") 
            metadata_table = dict(
                table_df=table_df,
                show_columns=show_columns,
                delta_columns="footprint(%),psnr,bpp,CR,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(","),
                formatter_dict={'size(byte)': "{:.0f}",'ssim':'{:.3e}'},
                extend_formatter_keys="footprint(%),psnr,bpp,CR,delta_psnr,delta_psnr(%),delta_bpp,delta_bpp(%)".split(",")
            )

            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            metadata_table['hide_index'] = hide_index
            metadata_table['cmap'] = cmap
            metadata_table['enable_cmap'] = enable_cmap
            metadata_table["show_deltas"] = show_deltas
            metadata_table["hide_date_init_from"] = hide_date_init_from
            resulting_table_df = create_comparing_results_table(**metadata_table)
            display(resulting_table_df)
        pass
    pass
