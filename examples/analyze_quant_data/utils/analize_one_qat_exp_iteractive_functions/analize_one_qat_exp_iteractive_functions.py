from utils.libs.std_python_libs import *
from utils.libs.data_science_libs import *
from utils.libs.graphics_and_interactive_libs import *
from utils.qat_features.create_comparing_qat_tables import create_comparing_results_table

from ipywidgets import Button, HBox, VBox

def pick_local_file_system_tmp_dataset(root_dir_tmp_datasets, QAT_DATASET_PATH):
    datasets_list = list(filter(lambda item: item.endswith(".csv"),os.listdir(root_dir_tmp_datasets)))
    QAT_DATASET_PATH = os.path.join(root_dir_tmp_datasets, datasets_list[0])
    @interact
    def pick_tmp_dataset(pick_dataset=datasets_list):
        QAT_DATASET_PATH = os.path.join(root_dir_tmp_datasets, pick_dataset)
        print(f"Dataset picked: {QAT_DATASET_PATH}")
        qat_df = pd.read_csv(QAT_DATASET_PATH)
        if "Unnamed: 0" in qat_df.columns:
            qat_df = qat_df.drop(["Unnamed: 0"], axis = 1)
            pass
        tabular_data = [
            ["Path", root_dir_tmp_datasets],
            ["Dataset Name", pick_dataset],
            ["Dataset Shape", qat_df.shape],
            ["Columns", str(list(qat_df.columns))],
            ["Quant Techs", str(list(set(qat_df["cmprss-class"])))],
            ["Quant Techs-2", str(list(set(qat_df["cmprss-class-2"])))],
        ]
        metadata_table = dict(
            tabular_data=tabular_data
        )
        table = tabulate.tabulate(**metadata_table)
        print(table)
        pass
    pass

def show_hist_or_pie_quant_tech(qat_df):
    # qat_df["cmprss-class-2"].iplot(kind="histogram", bins=20, theme="white", title="cmprss-class-2", xTitle='cmprss-class-2', yTitle='Count')
    templates_list = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    @interact
    def scatter_plot(hist_or_pie_chart=True, theme=list(cf.themes.THEMES.keys()),
                     colorscale=list(cf.colors._scales_names.keys()),
                    templates_pie=templates_list):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if hist_or_pie_chart:
                qat_df["cmprss-class-2"].iplot(
                    kind="histogram", bins=20,
                    title="cmprss-class-2", xTitle='cmprss-class-2', yTitle='Count',
                    theme=theme, colorscale=colorscale)
                pass
            else:
                occr_dict = dict(collections.Counter(qat_df["cmprss-class-2"].values))
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

def show_hist_or_pie_quant_tech_2(qat_df):
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
                qat_df["cmprss-class-3"].iplot(
                    kind="histogram", bins=20,
                    title="cmprss-class-3", xTitle='cmprss-class-3', yTitle='Count',
                    theme=theme, colorscale=colorscale)
                pass
            else:
                templates_list = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
                occr_dict = dict(collections.Counter(qat_df["cmprss-class-3"].values))
                keys, values = list(occr_dict.keys()), list(occr_dict.values())
                data = np.array(values)[:, np.newaxis]
                # .iplot(kind="pie", theme="white", y = "Occr")
                tmp_df = pd.DataFrame(data = data, index = keys, columns = ["Occr"])
                fig = px.pie(tmp_df, values='Occr', names=tmp_df.index, template = templates_pie)
                fig.show()
                pass
        pass
    pass

def show_scatter_plot(qat_df, jpeg_df):
    @interact
    def scatter_plot(flag_add_jpeg_data=False, x=list(qat_df.select_dtypes('number').columns),
                     y=list(qat_df.select_dtypes('number').columns)[1:],
                     theme=list(cf.themes.THEMES.keys()), 
                     colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if flag_add_jpeg_data:
                tmp_df = pd.concat([jpeg_df, qat_df[(qat_df["bpp"] <= 12.0)]], axis = 0)
                tmp_df.iplot(kind='scatter', x=x, y=y, mode='markers', 
                 xTitle=x.capitalize(), yTitle=y.capitalize(),
                 text='cmprss-class',
                 title=f'{y.capitalize()} vs {x.capitalize()} for QAT-Compression + Jpge',
                theme=theme, colorscale=colorscale, categories="cmprss-class-2")
            else:
                qat_df[(qat_df["bpp"] <= 12.0)].iplot(kind='scatter', x=x, y=y, mode='markers', 
                 xTitle=x.capitalize(), yTitle=y.capitalize(),
                 text='cmprss-class',
                 title=f'{y.capitalize()} vs {x.capitalize()} for QAT-Compression',
                theme=theme, colorscale=colorscale, categories="cmprss-class-2")
            pass
        pass
    pass

def show_scatter_plot_2(qat_df, jpeg_df):
    @interact
    def scatter_plot(flag_add_jpeg_data=False, x=list(qat_df.select_dtypes('number').columns),
                     y=list(qat_df.select_dtypes('number').columns)[1:],
                     theme=list(cf.themes.THEMES.keys()), 
                     colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if flag_add_jpeg_data:
                tmp_df = pd.concat([jpeg_df, qat_df[(qat_df["bpp"] <= 12.0)]], axis = 0)
                tmp_df.iplot(kind='scatter', x=x, y=y, mode='markers', 
                 xTitle=x.capitalize(), yTitle=y.capitalize(),
                 text='cmprss-class',
                 title=f'{y.capitalize()} vs {x.capitalize()} for QAT-Compression + Jpge',
                theme=theme, colorscale=colorscale, categories="cmprss-class-3")
            else:
                qat_df[(qat_df["bpp"] <= 12.0)].iplot(kind='scatter', x=x, y=y, mode='markers', 
                 xTitle=x.capitalize(), yTitle=y.capitalize(),
                 text='cmprss-class',
                 title=f'{y.capitalize()} vs {x.capitalize()} for QAT-Compression',
                theme=theme, colorscale=colorscale, categories="cmprss-class-3")
            pass
        pass
    pass

def show_box_plot(qat_df, jpeg_df):
    # qat_df[['cmprss-class-3', 'psnr']].pivot(columns='cmprss-class-3', values='psnr').iplot(kind='box')
    @interact
    def scatter_plot(flag_add_jpeg_data=False, theme=list(cf.themes.THEMES.keys()), colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if flag_add_jpeg_data:
                tmp_df = pd.concat([jpeg_df, qat_df[(qat_df["bpp"] <= 12.0)]], axis = 0)
                tmp_df[['cmprss-class-3', 'psnr']]\
                    .pivot(columns='cmprss-class-3', values='psnr')\
                    .iplot(kind='box', theme=theme, colorscale=colorscale, title = 'Psnr(db) for QAT-Compression + Jpeg')
            else:
                qat_df[['cmprss-class-3', 'psnr']]\
                    .pivot(columns='cmprss-class-3', values='psnr')\
                    .iplot(kind='box', theme=theme, colorscale=colorscale, title = 'Psnr(db) for QAT-Compression')
            pass
        pass
    pass

def show_box_plot_2(qat_df, jpeg_df):
    # qat_df[['cmprss-class-3', 'bpp']].pivot(columns='cmprss-class-3', values='bpp').iplot(kind='box')
    @interact
    def scatter_plot(flag_add_jpeg_data=False, theme=list(cf.themes.THEMES.keys()), colorscale=list(cf.colors._scales_names.keys())):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if flag_add_jpeg_data:
                tmp_df = pd.concat([jpeg_df, qat_df[(qat_df["bpp"] <= 12.0)]], axis = 0)
                tmp_df[['cmprss-class-3', 'bpp']].\
                    pivot(columns='cmprss-class-3', values='bpp').\
                    iplot(kind='box', theme=theme, colorscale=colorscale, title = 'Bpp for QAT-Compression + Jpeg')
            else:
                qat_df[['cmprss-class-3', 'bpp']]\
                    .pivot(columns='cmprss-class-3', values='bpp')\
                .iplot(kind='box', theme=theme, colorscale=colorscale, title = 'Bpp for QAT-Compression')
            pass
        pass
    pass

def create_resulting_table_for_comparison_reasons(table_df):
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
