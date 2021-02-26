from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

# ---------------------------------------------------- #
# Utils
# ---------------------------------------------------- #
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = float(np.round([height], 2)[0])
        if height == 0.0: continue
        ax.annotate('{}'.format(height), size=7,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation = 45)
        pass
    pass


def remap_quant_techs(a_qt):
    if not a_qt.startswith("QATRLQ"): return a_qt
    old_p = "QATRLQ,NNPCW,PCW".split(",")
    new_p = "RL,L,C".split(",")
    change_dict = dict(zip(old_p, new_p))
    a_qt = a_qt.split(":")
    new_qt: list = []
    for item in a_qt:
        if item in old_p:
            new_qt.append(change_dict[item])
        else:
            new_qt.append(item)
    return ':'.join(new_qt)


def chart_1(a_df: pd.DataFrame, meta_plot_dict: dict, ax) -> None:
    """Comment it."""

    group_attributes = meta_plot_dict["chart0"]["group_attributes"]
    clss_attr =  meta_plot_dict["chart0"]["clss_attr"]
    x_attr =  meta_plot_dict["chart0"]["x_attr"]
    # group_attributes: list = "quant_techs".split(",")    
    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))
    dfs_groups = a_tmp_df.groupby(by = group_attributes)
    dfs_groups.size()\
        .to_frame(name='counts')\
        .plot.pie(y='counts', autopct='%1.1f%%',ax=ax)
    ax.set_title(f'Freq. Trained Instances by {x_attr.upper()}', fontweight="bold")
    ax.set_ylabel('Freq. [%]')
    ax.set_xlabel(f'{x_attr.upper()}')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    pass


def chart_2(a_df: pd.DataFrame, meta_plot_dict: dict, ax) -> None:
    """Comment it."""
    # group_attributes: list = "nbits".split(",")
    group_attributes = meta_plot_dict["chart1"]["group_attributes"]
    clss_attr = group_attributes = meta_plot_dict["chart1"]["clss_attr"]
    x_attr = group_attributes = meta_plot_dict["chart1"]["x_attr"]
    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))

    dfs_groups = a_tmp_df.groupby(by = group_attributes)
    
    dfs_groups.size()\
        .to_frame(name='counts')\
        .plot.pie(y='counts', autopct='%1.1f%%',ax=ax)
    ax.set_title(f'Freq. Trained Instances by {x_attr.upper()}', fontweight="bold", )
    ax.set_ylabel('Freq. [%]')
    ax.set_xlabel(f"{x_attr.upper()}")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    pass


def chart_3(a_df: pd.DataFrame, meta_plot_dict: dict, ax) -> None:
    """Comment it."""

    group_attributes = meta_plot_dict["chart2"]["group_attributes"]
    clss_attr =  meta_plot_dict["chart2"]["clss_attr"]
    x_attr =  meta_plot_dict["chart2"]["x_attr"]
    target_attr =  meta_plot_dict["chart2"]["target_attr"]

    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))
    # group_attributes: list = "quant_techs,nbits".split(",")
    a_tmp_df = a_tmp_df.groupby(group_attributes)
    lr_dict = dict()
    # target_attr = 'psnr'
    for key, grp in a_tmp_df:
        qt, lr = key
        item = lr_dict.setdefault(qt, dict())
        val = grp.sort_values(by = [f"{target_attr}"], ascending = False)[f"{target_attr}"].values[0]
        item[lr] = val
        pass
    tmp_df = pd.DataFrame(list(lr_dict.values()), \
                index = list(lr_dict.keys()))
    bars = tmp_df.plot.bar(ax=ax, rot=30, width=0.9, edgecolor='black',)
    if x_attr.lower() == "bpp".lower() or x_attr.lower() == "CR".lower():
        all_xticks = ax.get_xticklabels()
        # for ii in range(len(all_xticks)-3):
        for ii in range(len(all_xticks)):
            xtick = all_xticks[ii]
            a_text = xtick.get_text()
            # if a_text.isdigit()and len(a_text) > 5:
            # if len(a_text) > 7:
            xtick.set_text(a_text[0:5])
            all_xticks[ii] = xtick
            pass
        ax.set_xticklabels(all_xticks)
        pass
    autolabel(rects=bars.patches, ax=ax)
    a_attr = group_attributes[1]
    ax.set_title(f'Freq. Trained Instances by {a_attr.upper()} for {x_attr.upper()}', fontweight="bold")
    ax.set_ylabel('Freq. [%]')
    ax.set_xlabel(f'{x_attr.upper()}')
    ax.grid(True)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    pass


def chart_4(
        a_df: pd.DataFrame, meta_plot_dict: dict,
        pruned_model_df: pd.DataFrame, tch_df: pd.DataFrame,
        tmp_jpeg_min_q: pd.DataFrame, tmp_jpeg_max_q:pd.DataFrame, ax) -> None:
    """Comment it."""

    group_attributes = meta_plot_dict["chart3"]["group_attributes"]
    clss_attr =  meta_plot_dict["chart3"]["clss_attr"]
    x_attr =  meta_plot_dict["chart3"]["x_attr"]
    target_attr =  meta_plot_dict["chart3"]["target_attr"]

    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))
    # group_attributes: list = "quant_techs,nbits".split(",")
    a_tmp_df = a_tmp_df.groupby(group_attributes)
    lr_dict = dict()
    # target_attr = 'psnr'
    for key, grp in a_tmp_df:
        qt, lr = key
        item = lr_dict.setdefault(lr, dict())
        val = grp.sort_values(by = [f"{target_attr}"], ascending = False)[f"{target_attr}"].values[0]
        item[qt] = val
        pass
    tmp_df = pd.DataFrame(list(lr_dict.values()), \
                index = list(lr_dict.keys())) # .sort_values(by = [x_attr])
    pruned_model_df.index = [0]

    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    tmp_df_2 = pd.concat([ tmp_jpeg_max_q[f"{target_attr}"], tmp_jpeg_min_q[f"{target_attr}"],
        tch_df[f"{target_attr}"], pruned_model_df[f"{target_attr}"],
    ], axis = 1, ignore_index = True)
    tmp_df_2.index = ["baselines"]
    pruned_class = pruned_model_df["cmprss-class"].values[0]
    bsln_class = ';'.join(tch_df["cmprss-class"].values[0].split(":")[1:])
    tmp_df_2.columns = f"jpeg({qmax}),jpeg({qmin}),BS-{bsln_class},{pruned_class}".upper().split(",")
    tmp_df = pd.concat([tmp_df, tmp_df_2], axis =0)
    bars = tmp_df.plot.bar(ax=ax, rot=30, width=1.1, edgecolor='black',)
    autolabel(rects=bars.patches, ax=ax)
 
    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]

    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")
    if target_attr == "psnr":
        ylabel = "psnr [db]"
    else:
        ylabel = target_attr
    ax.set_title(f'{ylabel.capitalize()} Values by {clss_attr.upper()} for each {x_attr.upper()}', fontweight="bold")
    ax.set_ylabel(f'{ylabel.capitalize()}')
    ax.set_xlabel(f'{x_attr.upper()}')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')

    if x_attr.lower() == "bpp" or x_attr.upper() == "CR":
        all_xticks = ax.get_xticklabels()
        # for ii in range(len(all_xticks)-3):
        for ii in range(len(all_xticks)):
            xtick = all_xticks[ii]
            a_text = xtick.get_text()
            if a_text.isnumeric() and len(a_text) > 5:
                xtick.set_text(a_text[0:5])
                all_xticks[ii] = xtick
                pass
            pass
        ax.set_xticklabels(all_xticks)
        pass
    
    indeces = a_df.index
    xlables = list(bars.get_xticklabels())



    rects = bars.patches
    labels = ["label%d" % i for i in range(len(rects))]
    for ii, (rect, label) in enumerate(zip(rects, labels)):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y() 
        if height == 0: continue
        # print(type(label), label)
        # jj = ii % len(xlables)
        # pprint(bars.get_xticklabels()[jj])
        # print(height, width, x, y)
        # ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='left', va='top', rotation=45)
        pass
    ax.grid(True)
    pass


def bar_chart(
        a_df: pd.DataFrame, meta_plot_dict: dict,
        pruned_model_df: pd.DataFrame, tch_df: pd.DataFrame,
        tmp_jpeg_min_q: pd.DataFrame, tmp_jpeg_max_q:pd.DataFrame, ax, pos) -> None:
    """Comment it."""

    group_attributes = meta_plot_dict[f"{pos}"]["group_attributes"]
    clss_attr =  meta_plot_dict[f"{pos}"]["clss_attr"]
    x_attr =  meta_plot_dict[f"{pos}"]["x_attr"]
    target_attr =  meta_plot_dict[f"{pos}"]["target_attr"]
    index =  meta_plot_dict[f"{pos}"]["index"]

    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))
    # group_attributes: list = "quant_techs,nbits".split(",")
    a_tmp_df = a_tmp_df.groupby(group_attributes)
    lr_dict = dict()
    # target_attr = 'psnr'
    for key, grp in a_tmp_df:
        qt, lr = key
        item = lr_dict.setdefault(lr, dict())
        val = grp.sort_values(by = [f"{target_attr}"], ascending = False)[f"{target_attr}"].values[0]
        item[qt] = val
        pass
    tmp_df = pd.DataFrame(list(lr_dict.values()), \
                index = list(lr_dict.keys())) # .sort_values(by = [x_attr])
    pruned_model_df.index = [0]

    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    tmp_df_2 = pd.DataFrame()
    if type(index) is list:
        matrix = np.ndarray(shape=(4,4))
        matrix[:,:] = np.nan
        x, y = np.arange(0, 4), np.arange(0, 4)
        vals = [tmp_jpeg_max_q[f"{target_attr}"], tmp_jpeg_min_q[f"{target_attr}"], tch_df[f"{target_attr}"], pruned_model_df[f"{target_attr}"]]
        for ii, (xx, yy) in enumerate(zip(x, y)):
            matrix[xx][yy] = vals[ii]
            pass
        pruned_class = pruned_model_df["cmprss-class"].values[0]
        bsln_class = ';'.join(tch_df["cmprss-class"].values[0].split(":")[1:])
        tmp_df_2 = pd.DataFrame(data = matrix, columns=f"jpeg({qmax}),jpeg({qmin}),BS-{bsln_class},{pruned_class}".upper().split(","))
        tmp_df_2.index = index
    else:
        # tmp_df_2 = pd.concat([ tmp_jpeg_max_q[f"{target_attr}"], tmp_jpeg_min_q[f"{target_attr}"],
        #     tch_df[f"{target_attr}"], pruned_model_df[f"{target_attr}"],
        # ], axis = 0, ignore_index = True)
        
        # tmp_df_2.index = [index] # ["baselines"]
        pruned_class = pruned_model_df["cmprss-class"].values[0]
        bsln_class = ';'.join(tch_df["cmprss-class"].values[0].split(":")[1:])
        # columns = f"jpeg({qmax}),jpeg({qmin}),BS-{bsln_class},{pruned_class}".split(",")
        # tmp_df_2.columns = columns

        max_index = max(tmp_df.index)
        
        if x_attr.lower() == "nbits" or x_attr.lower() == "lr":
            columns = f"jpeg({qmax}),jpeg({qmin}),BS-{bsln_class},{pruned_class}".upper().split(",")
            matrix = np.ndarray(shape=(2,4))
            matrix[:,:] = np.nan
            matrix[0,:2] = tmp_jpeg_max_q[f"{target_attr}"].values[0], tmp_jpeg_min_q[f"{target_attr}"].values[0]
            matrix[1,2:] = tch_df[f"{target_attr}"].values[0], pruned_model_df[f"{target_attr}"].values[0]
            if x_attr.lower() == "nbits":
                index = ["Jpeg", "32"]
            else:
                index = ["Jpeg", "1e-4"]
            tmp_df_2 = pd.DataFrame(data = matrix, columns=columns, index=[max_index+1, max_index+2])
            pass
        elif x_attr.lower() == "bpp":
            columns = f"jpeg({qmax}),jpeg({qmin}),{pruned_class},BS-{bsln_class}".upper().split(",")
            matrix = np.ndarray(shape=(4,4))
            matrix[:,:] = np.nan
            """
            matrix[0,:2] = tmp_jpeg_max_q[f"{target_attr}"].values[0], tmp_jpeg_min_q[f"{target_attr}"].values[0]
            matrix[1,2] = pruned_model_df[f"{target_attr}"].values[0]
            matrix[2,3] = tch_df[f"{target_attr}"].values[0]
            
            index = ["Jpeg", 
                "%.4f" % (pruned_model_df[f"bpp"].values[0],),
                "%.4f" % (tch_df[f"bpp"].values[0],),
            ]
            tmp_df_2 = pd.DataFrame(data = matrix, columns=columns, index=[max_index+1, max_index+2, max_index+3])
            """
            values = [
                tmp_jpeg_max_q[f"{target_attr}"].values[0],
                tmp_jpeg_min_q[f"{target_attr}"].values[0],
                pruned_model_df[f"{target_attr}"].values[0],
                tch_df[f"{target_attr}"].values[0],
            ]
            matrix[range(0, 4), range(0, 4)] = values
            index = [
                tmp_jpeg_max_q[f"bpp"].values[0],
                tmp_jpeg_min_q[f"bpp"].values[0],
                pruned_model_df[f"bpp"].values[0],
                tch_df[f"bpp"].values[0],
            ]
            tmp_df_2 = pd.DataFrame(data = matrix, columns=columns, index=index)
            pass
        pass
    
    # tmp_df = pd.concat([tmp_df, tmp_df_2], axis =0)
    values = [
        tmp_jpeg_max_q[f"{target_attr}"].values[0],
        tmp_jpeg_min_q[f"{target_attr}"].values[0],
        pruned_model_df[f"{target_attr}"].values[0],
        tch_df[f"{target_attr}"].values[0],
    ]
    columns = f"jpeg({qmax}),jpeg({qmin}),{pruned_class},BS-{bsln_class}".upper().split(",")
    a_matrix = np.zeros(shape=(tmp_df.shape[0],4))
    a_matrix[:,:] = values
    a_matrix_df = pd.DataFrame(data=a_matrix, columns=columns)
    a_matrix_df.index = tmp_df.index
    old_cls = tmp_df.columns

    tmp_df = pd.concat([tmp_df, a_matrix_df], axis = 1, ignore_index=False)
    tmp_df.columns = list(old_cls) + list(a_matrix_df.columns)
    # print(tmp_df.shape)
    # print(tmp_df.head(5))
    # sys.exit(0)
    tmp_df.index = list(lr_dict.keys())
    
    # bars = tmp_df.sort_index().plot.bar(ax=ax, rot=30)
    bars = tmp_df.plot.bar(ax=ax, rot=30, width=0.9, edgecolor='black',)
    # bars_2 = tmp_df_2.plot.bar(ax=ax, rot=30)
    all_xticks = list(ax.get_xticklabels())

    """
    if x_attr == "nbits" or x_attr == "lr":

        # a_index = np.sort(np.array(tmp_df.index))
        x = (len(all_xticks) - 2 + len(all_xticks) - 1) // 2
        
        ylim = ax.get_ylim()
        ax.vlines(x=x, ymin=ylim[0], ymax=ylim[1], linestyle="-.", color='black')
        ax.text(x=x, y=np.mean(ylim), s="Right-Refs.", fontweight="bold")

        xtick = all_xticks[-1]
        xtick.set_text(index[-1])
        all_xticks[-1] = xtick
        xtick = all_xticks[-2]
        xtick.set_text(index[-2])
        all_xticks[-2] = xtick
        pass
    """
    if x_attr.lower() == "bpp".lower() or x_attr.lower() == "CR".lower():

        # for ii in range(len(all_xticks)-3):
        for ii in range(len(all_xticks)):
            xtick = all_xticks[ii]
            a_text = xtick.get_text()
            # if a_text.isdigit() and len(a_text) > 5:
                # if len(a_text) > 7:
            xtick.set_text(a_text[0:5])
            all_xticks[ii] = xtick
            pass
        ylim = ax.get_ylim()
        ax.vlines(x=9.5, ymin=ylim[0], ymax=ylim[1], linestyle="-.", color='black')
        ax.text(x=9.5, y=np.mean(ylim), s="thld-BPP(Image):8", fontweight="bold")

        """
        xtick = all_xticks[-1]
        xtick.set_text(index[-1])
        all_xticks[-1] = xtick

        xtick = all_xticks[-2]
        xtick.set_text(index[-2])
        all_xticks[-2] = xtick

        xtick = all_xticks[-3]
        xtick.set_text(index[-3])
        all_xticks[-3] = xtick
        """
        pass
    ax.set_xticklabels(all_xticks)

    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]

    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")
    if target_attr == "psnr":
        ylabel = "psnr [db]"
    else:
        ylabel = target_attr
    ax.set_title(f'{ylabel.capitalize()} Values by {clss_attr.upper()} for each {x_attr.upper()}', fontweight="bold")
    ax.set_ylabel(f'{ylabel.capitalize()}')
    ax.set_xlabel(f'{x_attr.upper()}')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')

    indeces = a_df.index
    xlables = list(bars.get_xticklabels())

    rects = bars.patches
    autolabel(rects=bars.patches, ax=ax)
    ax.grid(True)
    pass

# ---------------------------------------------------- #
# Chart: create_2pies_2bars_chart
# ---------------------------------------------------- #

def create_2pies_2bars_chart(a_df, tch_df, pruned_model_df, tmp_jpeg_min_q, tmp_jpeg_max_q, meta_plot_dict: dict, show_fig: bool = False):
    """Comment it.
    Returns:
    --------
    `fig` - if empty dataset, i.e., a_df.shape.[0] == 0, fig will be None\n
    `ax` - if empty dataset, i.e., a_df.shape.[0] == 0, ax will be None\n
    """

    if a_df.shape[0] == 0: return None, None

    root_dir = meta_plot_dict["root_dir"]
    # print(root_dir)
    try:
        os.makedirs(root_dir)
    except Exception as err:
        # print(err)
        pass

    fig, axes = plt.subplots(2,2,figsize=(25,10))
    tch = tch_df["cmprss-class"].values[0]
    fig.suptitle(f"Quant Trained Models Overview\nwith fixed {tch}", fontweight="bold",  fontsize=20)
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax=axes[0]
    chart_1(a_df=a_df, meta_plot_dict=meta_plot_dict, ax=ax) 

    ax=axes[1]
    chart_2(a_df=a_df, meta_plot_dict=meta_plot_dict, ax=ax)


    ax=axes[2]
    chart_3(a_df=a_df, meta_plot_dict=meta_plot_dict, ax=ax) 


    ax = axes[3]
    chart_4(
        a_df, meta_plot_dict,
        pruned_model_df, tch_df,
        tmp_jpeg_min_q, tmp_jpeg_max_q, ax)

    img_name = meta_plot_dict["img_name"]
    # img_name = "pie_plus_bar_wrt_lr_qatrlq_5.png"
    # root_dir = os.path.join("/home/franec94/Documents/tmp_images_from_notebooks")
    try: os.makesdir(root_dir)
    except: pass
    plt.savefig(os.path.join(root_dir, img_name))
    if show_fig:
        plt.show()
    return fig, axes

# ---------------------------------------------------- #
# Chart: create_grid_chart_4bars
# ---------------------------------------------------- #

def create_grid_chart_4bars(
    a_df, tch_df, pruned_model_df, tmp_jpeg_min_q, tmp_jpeg_max_q, meta_plot_dict: dict,  show_fig: bool= False) -> None:
    """Comment it."""

    if a_df.shape[0] == 0: return

    fig, axes = plt.subplots(2,2,figsize=(25,10))
    tch = tch_df["cmprss-class"].values[0]

    root_dir = meta_plot_dict["root_dir"]
    try: os.makedirs(root_dir)
    except: pass

    fig.suptitle(f"Quant Trained Models Overview,\nwith fixed {tch}", fontweight="bold",  fontsize=20)
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    charts_list = meta_plot_dict["charts_list"]
    for ii, a_chart_dict in enumerate(charts_list):
        ax = axes[ii]
        pos = list(a_chart_dict.keys())[0]
        bar_chart(
            a_df, a_chart_dict,
            pruned_model_df, tch_df,
            tmp_jpeg_min_q, tmp_jpeg_max_q, ax, pos)
    
    img_name = meta_plot_dict["img_name"]
    plt.savefig(os.path.join(root_dir, img_name))
    if show_fig:
        plt.show()
    pass

# ---------------------------------------------------- #
# Chart: create_grid_chart_3bars_1scatter
# ---------------------------------------------------- #


def create_scatterplot(a_df, meta_plot_dict, pruned_model_df, tch_df, tmp_jpeg_min_q, tmp_jpeg_max_q, ax, siren_df = pd.DataFrame(), jpeg_df = pd.DataFrame()):
    """Commnet it."""

    pos="scatter"

    group_attributes = meta_plot_dict[f"{pos}"]["group_attributes"]
    clss_attr =  meta_plot_dict[f"{pos}"]["clss_attr"]
    x_attr =  meta_plot_dict[f"{pos}"]["x_attr"]
    target_attr =  meta_plot_dict[f"{pos}"]["target_attr"]

    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{clss_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{clss_attr}"].values))
    # group_attributes: list = "quant_techs,nbits".split(",")
    a_tmp_df = a_tmp_df.groupby(group_attributes)
    
    # target_attr = 'psnr'
    pdfs_list : list = []
    for key, grp in a_tmp_df:
        val = grp.sort_values(by = [f"{target_attr}"], ascending = False).iloc[0]
        pdfs_list.append(val)
        pass
    
    
    tmp_df = pd.DataFrame(pdfs_list)
    tmp_df_bu = pd.DataFrame(tmp_df)
    pruned_model_df.index = [0]

    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    pick_columns = f"{target_attr},{x_attr},cmprss-class".split(",")
    tmp_df_2 = pd.concat([tmp_jpeg_max_q[pick_columns], tmp_jpeg_min_q[pick_columns],
        tch_df[pick_columns], pruned_model_df[pick_columns],
    ], axis = 0, ignore_index = True)
    tmp_df_2.index = ["baselines"] * tmp_df_2.shape[0]
    pruned_class = pruned_model_df["cmprss-class"].values[0]
    bsln_class = ';'.join(tch_df["cmprss-class"].values[0].split(":")[1:])
    tmp_df_2.columns = f"{target_attr},{x_attr},quant_techs".split(",")
    
    pick_columns = f"{target_attr},{x_attr},quant_techs".split(",")
    tmp_df = pd.concat([tmp_df[pick_columns], tmp_df_2[pick_columns]], axis =0)
    
    # tmp_df.plot.scatter(x=x_attr, y=target_attr, ax=ax)
    colors = sns.color_palette()
    unique_qt = sorted(list(tmp_df["quant_techs"].unique())) + ['SIREN']
    custom_colors = colors[0:len(unique_qt)][::-1]
    colors_dict = dict(zip(unique_qt, custom_colors))
    
    # sns.scatterplot(data=tmp_df, x=x_attr, y=target_attr, ax=ax, hue="quant_techs", marker='x', palette=colors_dict)
    if siren_df.shape[0] != 0:
        pos = siren_df["bpp"] >= 8
        # sns.scatterplot(data=siren_df[siren_df["bpp"] > 8], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='>', palette=colors_dict)
        label = siren_df[pos]["cmprss-class-2"].values[0]
        ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
            label=f"{label}-o.p.", marker='^', color=custom_colors[-1], edgecolor='black', alpha=0.7)

        pos = (siren_df["bpp"] <= 8) & (siren_df["psnr"] >= tmp_jpeg_min_q["psnr"].values[0])
        # sns.scatterplot(data=siren_df[pos], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='o', palette=colors_dict)
        label = siren_df[pos]["cmprss-class-2"].values[0]
        ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
            label=f"{label}-c.g.", marker='+', color=custom_colors[-1], edgecolor='black', alpha=0.7)

        pos = (siren_df["bpp"] <= 8) & (siren_df["psnr"] < tmp_jpeg_min_q["psnr"].values[0])
        # sns.scatterplot(data=siren_df[pos], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='<', palette=colors_dict)
        label = siren_df[pos]["cmprss-class-2"].values[0]
        ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
            label=f"{label}-u.p.", marker='v', color=custom_colors[-1], edgecolor='black', alpha=0.7)
        pass

    if jpeg_df.shape[0] != 0:
        label = jpeg_df["cmprss-class-2"].values[0]
        ax.scatter(x=jpeg_df[x_attr].values, y=jpeg_df[target_attr].values,
            label=f"{label}", marker='v', color=custom_colors[-2], edgecolor='black', alpha=0.7)
        pass
    
    label = tmp_jpeg_max_q["cmprss-class"].values[0]
    ax.scatter(x=tmp_jpeg_max_q["bpp"].values[0], y=tmp_jpeg_max_q["psnr"].values[0], marker='p', s=100, color=colors_dict[f"{label}"], label=f"{label}", edgecolor='black',)
    x=tmp_jpeg_max_q["bpp"].values[0]; y=tmp_jpeg_max_q["psnr"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tmp_jpeg_max_q, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='^', s = 100, palette=colors_dict)

    label = tmp_jpeg_min_q["cmprss-class"].values[0]
    ax.scatter(x=tmp_jpeg_min_q["bpp"].values[0], y=tmp_jpeg_min_q["psnr"].values[0], marker='p', s=100, color=colors_dict[f"{label}"], label=f"{label}", edgecolor='black',)
    x=tmp_jpeg_min_q["bpp"].values[0]; y=tmp_jpeg_min_q["psnr"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tmp_jpeg_min_q, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='^', s = 100, palette=colors_dict)

    label = tch_df["cmprss-class"].values[0]
    ax.scatter(x=tch_df["bpp"].values[0], y=tch_df["psnr"].values[0], marker='*', s=100, color=colors_dict[f"{label}"], label=f"{label}", edgecolor='black',)
    x=tch_df["bpp"].values[0]; y=tch_df["psnr"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tch_df, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='*', s = 150, palette=colors_dict)
    
    label = pruned_model_df["cmprss-class"].values[0]
    ax.scatter(x=pruned_model_df["bpp"].values[0], y=pruned_model_df["psnr"].values[0], marker='D', s=50, color=colors_dict[f"{label}"], label=f"{label}", edgecolor='black',)
    x=pruned_model_df["bpp"].values[0]; y=pruned_model_df["psnr"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=pruned_model_df, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='d', s = 100, palette=colors_dict)

    # label = tmp_df_bu["quant_techs"].values[0]
    # ax.scatter(x=tmp_df_bu["bpp"].values, y=tmp_df_bu["psnr"].values, marker='d', s=100, color=colors_dict[f"{label}"], label=f"{label}")
    sns.scatterplot(data=tmp_df_bu, x=x_attr, y=target_attr, ax=ax, hue="quant_techs", marker='x', s = 125, palette=colors_dict)
    max_qt = tmp_df_bu.sort_values(by=["psnr"], ascending=False).iloc[0,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt["bpp"].values[0]; y=max_qt["psnr"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')        

    max_qt = tmp_df_bu.sort_values(by=["psnr"], ascending=False).iloc[2,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt["bpp"].values[0]; y=max_qt["psnr"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='left', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')     

    max_qt = tmp_df_bu[tmp_df_bu["nbits"] == 8].sort_values(by=["psnr"], ascending=False).iloc[0,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt["bpp"].values[0]; y=max_qt["psnr"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='left', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')     

    

    ax.legend(title="Quant/Pruned Techs. + BS and Jpeg")
 
    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]

    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")
    ylim = ax.get_ylim()
    ax.vlines(x=8.0, ymin=ylim[0], ymax=ylim[1], linestyle=":")
    ax.text(y=np.mean(ylim), x=8, s=f"thld-BPP(Image):8", fontweight="bold")
    if target_attr == "psnr":
        ylabel = "psnr [db]"
    else:
        ylabel = target_attr
        pass

    ax.set_title(f'{ylabel.capitalize()} Values by {clss_attr.upper()} for each {x_attr.upper()}', fontweight="bold")
    ax.set_ylabel(f'{ylabel.capitalize()}')
    ax.set_xlabel(f'{x_attr.upper()}')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    pass


def create_grid_chart_3bars_1scatter(
    a_df, tch_df, pruned_model_df,
    tmp_jpeg_min_q, tmp_jpeg_max_q,
    meta_plot_dict: dict, show_fig: bool = False,
    siren_bsln_df = pd.DataFrame(), jpeg_df = pd.DataFrame()) -> None:
    """Comment it."""

    # fig, axes = plt.subplots(2,2,figsize=(25,10),  constrained_layout=True, gridspec_kw={'wspace': 0.2, 'hspace': 0.01, 'vspace': 0.2})

    if a_df.shape[0] == 0: return # None, None

    root_dir = meta_plot_dict["root_dir"]
    try: os.makedirs(root_dir)
    except: pass

    fig, axes = plt.subplots(2,2,figsize=(25,10))
    tch = tch_df["cmprss-class"].values[0]
    fig.suptitle(f"Quant Trained Models Overview,\nwith {tch}", fontweight="bold",  fontsize=20)
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    tch = tch_df["cmprss-class"].values[0]
    fig.suptitle(f"Quant Trained Models Overview,\nwith fixed {tch}", fontweight="bold",  fontsize=20)
    # if axes.shape[0] > 1: axes = axes.flatten(); pass

    charts_list = meta_plot_dict["charts_list"]
    charts_pos = "0,2,1,3".split(",")
    charts_pos = list(map(int, charts_pos))
    # for ii, a_chart_dict in enumerate(charts_list):
    for ii, a_chart_dict in zip(charts_pos, charts_list):
        ax = axes[ii]
        pos = list(a_chart_dict.keys())[0]
        bar_chart(
            a_df, a_chart_dict,
            pruned_model_df, tch_df,
            tmp_jpeg_min_q, tmp_jpeg_max_q, ax, pos)
        pass
    
    ax=axes[-1]
    create_scatterplot(
        a_df, meta_plot_dict,
        pruned_model_df, tch_df,
        tmp_jpeg_min_q, tmp_jpeg_max_q, ax,
        siren_df=siren_bsln_df, jpeg_df=jpeg_df)
    img_name = meta_plot_dict["img_name"]
    plt.savefig(os.path.join(root_dir, img_name))
    if show_fig:
        plt.show()
        pass
    
    # plt.subplots_adjust(hspace = 0.5, bottom=0.1, right=0.8, top=0.9)

    fig, ax = plt.subplots(1,1, figsize=(7,5))
    create_scatterplot(
        a_df, meta_plot_dict,
        pruned_model_df, tch_df,
        tmp_jpeg_min_q, tmp_jpeg_max_q, ax,
        siren_df=siren_bsln_df, jpeg_df=jpeg_df)
    img_name = "scatter_plot_psnr_vs_bpp.png"
    plt.savefig(os.path.join(root_dir, img_name))

    pass

# ---------------------------------------------------- #
# Chart: create_grid_chart_3bars_1scatter_v2
# ---------------------------------------------------- #


def add_coordinates(
    tmp_df_bu: pd.DataFrame, tch_df : pd.DataFrame = pd.DataFrame(),
    tmp_jpeg_max_q: pd.DataFrame = pd.DataFrame(), tmp_jpeg_min_q: pd.DataFrame = pd.DataFrame(),
    pruned_model_df: pd.DataFrame = pd.DataFrame(),
    x_attr: str = "bpp", y_attr : str = "psnr",
    ax = None) -> None:

    """Comment it."""

    # Add data point for Jpeg max arbitrary chosen quality.
    # label = tmp_jpeg_max_q["cmprss-class"].values[0]
    x=tmp_jpeg_max_q[f"{x_attr}"].values[0]
    y=tmp_jpeg_max_q[f"{y_attr}"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tmp_jpeg_max_q, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='^', s = 100, palette=colors_dict)

    # Add data point for Jpeg min arbitrary chosen quality.
    # label = tmp_jpeg_min_q["cmprss-class"].values[0]
    x=tmp_jpeg_min_q[f"{x_attr}"].values[0]
    y=tmp_jpeg_min_q[f"{y_attr}"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tmp_jpeg_min_q, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='^', s = 100, palette=colors_dict)

    # Add data point for Baseline Model.
    # label = tch_df["cmprss-class"].values[0]
    x=tch_df[f"{x_attr}"].values[0]
    y=tch_df[f"{y_attr}"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=tch_df, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='*', s = 150, palette=colors_dict)
    
    # Add data point for Pruned Baseline Model.
    # label = pruned_model_df["cmprss-class"].values[0]
    x=pruned_model_df[f"{x_attr}"].values[0]
    y=pruned_model_df[f"{y_attr}"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    # sns.scatterplot(data=pruned_model_df, x=x_attr, y=target_attr, ax=ax, hue="cmprss-class", marker='d', s = 100, palette=colors_dict)

    # Add quant examples data points.
    # label = tmp_df_bu["quant_techs"].values[0]
    # ax.scatter(x=tmp_df_bu[f"{x_attr}"].values, y=tmp_df_bu[f"{y_attr}"].values, marker='d', s=100, color=colors_dict[f"{label}"], label=f"{label}")
    # sns.scatterplot(data=tmp_df_bu, x=x_attr, y=target_attr, ax=ax, hue="quant_techs", marker='x', s = 125, palette=colors_dict)
    max_qt = tmp_df_bu.sort_values(by=[f"{y_attr}"], ascending=False).iloc[0,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt[f"{x_attr}"].values[0]
    y=max_qt[f"{y_attr}"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y+1.5, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='center', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')        

    max_qt = tmp_df_bu.sort_values(by=[f"{y_attr}"], ascending=False).iloc[2,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt[f"{x_attr}"].values[0]
    y=max_qt[f"{y_attr}"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='left', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')     

    max_qt = tmp_df_bu[tmp_df_bu["nbits"] == 8].sort_values(by=[f"{y_attr}"], ascending=False).iloc[0,:]
    max_qt = pd.DataFrame(data=[max_qt.values], columns=max_qt.index)
    x=max_qt[f"{x_attr}"].values[0]
    y=max_qt[f"{y_attr}"].values[0]
    nbits = max_qt["nbits"].values[0]
    ax.text(x=x, y=y, s=f"({x:.2f};{y:.2f})-bits{nbits:.0f}", fontweight="bold", size = 7,
        horizontalalignment='left', verticalalignment='center',)
    ax.scatter(x=x, y=y, marker=".", color='red')   
    pass


def create_a_scatter_plot(
    tmp_df: pd.DataFrame, a_df: pd.DataFrame,
    tch_df: pd.DataFrame, pruned_model_df: pd.DataFrame,
    tmp_jpeg_max_q:pd.DataFrame = pd.DataFrame(), tmp_jpeg_min_q:pd.DataFrame = pd.DataFrame(),
    jpeg_df:pd.DataFrame = pd.DataFrame(), siren_df:pd.DataFrame = pd.DataFrame(),
    x_attr: str = "bpp", y_attr : str = "psnr",
    ax = None, show_plot: bool = False) -> None:
    """Comment it."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        pass

    ax.set_xlabel("BPP", fontweight = "bold")
    ax.set_ylabel("Psnr [db]", fontweight = "bold")
    
    ax.scatter(x=jpeg_df[f"{x_attr}"], y=jpeg_df[f"{y_attr}"], marker = "x", s=50, label = "jpeg")
    # ax.scatter(x=bs_tmp[f"{x_attr}"], y=bs_tmp["psnr"], marker = "x", s=50, label = "SIREN")

    # sns.scatterplot(data=tmp_df, x=x_attr, y=target_attr, ax=ax, hue="quant_techs", marker='x', palette=colors_dict)
    # siren_df = bs_tmp
    target_attr = 'psnr'
    if siren_df.shape[0] != 0:
        pos = siren_df[f"{x_attr}"] >= 8
        # sns.scatterplot(data=siren_df[siren_df[f"{x_attr}"] > 8], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='>', palette=colors_dict)
        label = siren_df[pos]["cmprss-class-2"].values[0]
        ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
            label=f"{label}-o.p.", marker='^', edgecolor='black', alpha=0.7)
        
        try:
            pos = (siren_df[f"{x_attr}"] <= 8) & (siren_df[f"{y_attr}"] >= tmp_jpeg_min_q[f"{y_attr}"].values[0])
            # sns.scatterplot(data=siren_df[pos], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='o', palette=colors_dict)
            label = siren_df[pos]["cmprss-class-2"].values[0]
            ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
                label=f"{label}-c.g.", marker='+', edgecolor='black', alpha=0.7)
        except: pass

        pos = (siren_df[f"{x_attr}"] <= 8) & (siren_df[f"{y_attr}"] < tmp_jpeg_min_q[f"{y_attr}"].values[0])
        # sns.scatterplot(data=siren_df[pos], x=x_attr, y=target_attr, ax=ax, hue="cmprss-class-2", marker='<', palette=colors_dict)
        label = siren_df[pos]["cmprss-class-2"].values[0]
        ax.scatter(x=siren_df[pos][x_attr].values, y=siren_df[pos][target_attr].values,
            label=f"{label}-u.p.", marker='v', edgecolor='black', alpha=0.7)
        pass

    sns.scatterplot(y='psnr', x=f"{x_attr}", data = a_df,
        ax = ax, s = 100, marker = 'd', hue = "quant_techs", edgecolor='black',)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')

    # Add THLD-Jpeg
    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list
    jmin_psnr = tmp_jpeg_min_q[f"{y_attr}"].values[0]
    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    # ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")
    ax.text(x=xlim[1], y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")

    ylim = ax.get_ylim()
    ax.vlines(x=8, ymin=ylim[0], ymax=ylim[1], linestyle=":")
    ax.text(y=np.mean(ylim), x=8, s=f"thld-BPP(Image):8", fontweight="bold")
    
    tmp_df_bu = tmp_df
    add_coordinates(tmp_df_bu, tch_df, tmp_jpeg_max_q, tmp_jpeg_min_q, pruned_model_df, x_attr=x_attr, y_attr=y_attr, ax=ax)
    pass


def plot_bars_lr_vs_psnr(merged_df:pd.DataFrame = pd.DataFrame(),
    tmp_jpeg_max_q: pd.DataFrame= pd.DataFrame(),
    tmp_jpeg_min_q: pd.DataFrame= pd.DataFrame(),
    ax = None,
    show_plot: bool = False) -> None:
    """Comment it."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    bars = sns.barplot(x="lr", y="psnr", data=merged_df, \
        hue="quant_techs", ax = ax, edgecolor='black',)
    
    ax.set_xlabel("LR", fontweight = "bold")
    ax.set_ylabel("Psnr [db]", fontweight = "bold")

    # Add THLD-Jpeg
    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]
    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")

    xticks_list = list(bars.get_xticklabels())
    for ii, item in enumerate(xticks_list):
        item.set_rotation(45)
        xticks_list[ii] = item
        pass
    bars.set_xticklabels(xticks_list)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    
    autolabel(rects=bars.patches, ax=ax)
    
    if show_plot:
        plt.show()
    pass


def plot_bars_bpp_vs_psnr(merged_df:pd.DataFrame = pd.DataFrame(),
    tmp_jpeg_max_q: pd.DataFrame= pd.DataFrame(),
    tmp_jpeg_min_q: pd.DataFrame= pd.DataFrame(),
    ax = None,
    show_plot: bool = False) -> None:
    """Comment it."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    bars = bars = sns.barplot(x="bpp", y="psnr", data=merged_df,
        hue="quant_techs", ax = ax, edgecolor='black',)

    ax.set_xlabel("BPP", fontweight = "bold")
    ax.set_ylabel("Psnr [db]", fontweight = "bold")

    # Add THLD-Jpeg
    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]
    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    # ax.text(x=np.mean(xlim), y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")
    ax.text(x=xlim[0], y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")

    xticks_list = list(bars.get_xticklabels())
    for ii, item in enumerate(xticks_list):
        item.set_rotation(45)
        text = item.get_text()
        item.set_text(text[0:4])
        xticks_list[ii] = item
        pass
    bars.set_xticklabels(xticks_list)

    ylim = ax.get_ylim()
    ax.vlines(x=5.7, ymin=ylim[0], ymax=ylim[1], linestyle=":")
    ax.text(y=np.mean(ylim), x=5.7, s=f"thld-BPP(Image):8", fontweight="bold")

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    autolabel(rects=bars.patches, ax=ax)

    if show_plot:
        plt.show()
    pass


def plot_bars_nbits_vs_psnr(merged_df:pd.DataFrame = pd.DataFrame(),
    tmp_jpeg_max_q: pd.DataFrame= pd.DataFrame(),
    tmp_jpeg_min_q: pd.DataFrame= pd.DataFrame(),
    ax = None,
    show_plot: bool = False) -> None:
    """Comment it."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        pass
    bars = sns.barplot(x="nbits", y="psnr", data=merged_df,
        hue="quant_techs", ax = ax, edgecolor='black',)

    ax.set_xlabel("NBITS", fontweight = "bold")
    ax.set_ylabel("Psnr [db]", fontweight = "bold")

    # Add THLD-Jpeg
    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list

    jmin_psnr = tmp_jpeg_min_q["psnr"].values[0]
    xlim = ax.get_xlim()
    ax.hlines(y=jmin_psnr, xmin=xlim[0], xmax=xlim[1], linestyle=":")
    ax.text(x=xlim[0], y=jmin_psnr+0.5, s=f"thld: jpeg{qmin}", fontweight="bold")

    xticks_list = list(bars.get_xticklabels())
    for ii, item in enumerate(xticks_list):
        item.set_rotation(45)
        xticks_list[ii] = item
        pass
    bars.set_xticklabels(xticks_list)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    
    autolabel(rects=bars.patches, ax=ax)
    if show_plot:
        plt.show()
    pass


def create_grid_chart_3bars_1scatter_v2(
    a_df: pd.DataFrame,
    tch_df: pd.DataFrame, pruned_model_df: pd.DataFrame,
    tmp_jpeg_max_q:pd.DataFrame = pd.DataFrame(), tmp_jpeg_min_q:pd.DataFrame = pd.DataFrame(),
    jpeg_df:pd.DataFrame = pd.DataFrame(), siren_df:pd.DataFrame = pd.DataFrame(),
    x_attr: str = "bpp", y_attr : str = "psnr",
    show_plot: bool = False, plot_confs: dict = None
    ) -> None:
    
    """Comment it."""

    fig, axes = plt.subplots(2, 2, figsize = (20, 10))
    tch = tch_df["cmprss-class"].values[0]
    fig.suptitle(f"Show Quant. Techs. Bar & Scatter Charts\nwith {tch}", fontweight="bold",  fontsize=20)
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    picked_cols = "lr,bpp,nbits,psnr,quant_techs".split(",")
    merged_df = pd.concat(
        [
            a_df[picked_cols],
            pruned_model_df[picked_cols],
            tch_df[picked_cols],
            tmp_jpeg_max_q[picked_cols],
            tmp_jpeg_min_q[picked_cols],
        ],
        axis = 0, ignore_index = True
    )
    merged_df.columns = picked_cols

    axes[0].set_title("Psnr [db] v.s. LR Bar Chart", fontweight = "bold")
    plot_bars_lr_vs_psnr(
        merged_df=merged_df,
        tmp_jpeg_max_q=tmp_jpeg_max_q,
        tmp_jpeg_min_q=tmp_jpeg_min_q,
        ax = axes[0])
    
    axes[1].set_title("Psnr [db] v.s. Bpp Bar Chart", fontweight = "bold")
    plot_bars_bpp_vs_psnr(
        merged_df=merged_df,
        tmp_jpeg_max_q=tmp_jpeg_max_q,
        tmp_jpeg_min_q=tmp_jpeg_min_q,
        ax = axes[1])
    
    axes[2].set_title("Psnr [db] v.s. NBITS Bar Chart", fontweight = "bold")
    plot_bars_nbits_vs_psnr(
        merged_df=merged_df,
        tmp_jpeg_max_q=tmp_jpeg_max_q,
        tmp_jpeg_min_q=tmp_jpeg_min_q,
        ax = axes[2])

    ax = axes[3]
    axes[3].set_title("Psnr [db] v.s. BPP Scatter Chart", fontweight = "bold")
    create_a_scatter_plot(
        a_df[picked_cols],
        a_df = merged_df, tch_df = tch_df, pruned_model_df = pruned_model_df,
        tmp_jpeg_max_q = tmp_jpeg_max_q, tmp_jpeg_min_q = tmp_jpeg_min_q,
        jpeg_df = jpeg_df, siren_df = siren_df,
        ax = ax, show_plot = False) 

    if plot_confs:
        root_dir = plot_confs["root_dir"]   # "/home/franec94/Documents/tmp_images_from_notebooks"
        try: os.makedirs(root_dir)
        except: pass
        file_name = plot_confs["file_name"] # "tmp_grid_3bars_1scatter.png"
        full_name = os.path.join(root_dir, file_name)
        plt.savefig(f"{full_name}")
        pass
    if show_plot:
        plt.show()
    pass


def create_bar_plot(
    a_df:pd.DataFrame, tmp_df:pd.DataFrame,
    pruned_model_df:pd.DataFrame, tch_df:pd.DataFrame,
    tmp_jpeg_min_q:pd.DataFrame, tmp_jpeg_max_q:pd.DataFrame,
    meta_plot_dict: dict = None, ax = None,
    siren_df = pd.DataFrame(), jpeg_df = pd.DataFrame(),
    x: str = "bpp", target_attr: str = "psnr",
    show_fig: bool = False, save_fig: bool = False,
    ) -> None:
    """Comment it."""

    values: list = [
        tmp_jpeg_max_q[f"{target_attr}"].values[0],
        tmp_jpeg_min_q[f"{target_attr}"].values[0],
        pruned_model_df[f"{target_attr}"].values[0],
        tch_df[f"{target_attr}"].values[0]
    ]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        pass
    colors = sns.color_palette()


    # tmp_df = copy.deepcopy(a_df)
    
    tmp_df["lr"].unique()
    lr_data_unique = tmp_df["quant_techs"].unique()
    color_dict: dict = dict(zip(lr_data_unique, colors[0:len(lr_data_unique)]))

    # bars = sns.barplot(x="nbits", y="psnr", data=tmp_df, hue="quant_techs", ax = ax)
    pos = a_df["psnr"] > 20.
    # bars = sns.barplot(x="CR", y="psnr", data=a_df, hue="quant_techs", ax = ax)
    bars = sns.barplot(x=f"{x}", y=f"{target_attr}", data=a_df[pos], hue="quant_techs", ax = ax)
    all_xticks = list(ax.get_xticklabels())

    k = values
    x = bars.patches[-1].get_x() + 1.0
    xx = 0.1
    width=0.266667
    q_list = [tmp_jpeg_max_q["cmprss-class"].values[0], tmp_jpeg_min_q["cmprss-class"].values[0]]
    q_list = [ii.split(":")[1] for ii in q_list]
    qmax, qmin = q_list
    pruned_class = pruned_model_df["cmprss-class"].values[0]
    bsln_class = ';'.join(tch_df["cmprss-class"].values[0].split(":")[1:])
    labels = f"jpeg({qmax}),jpeg({qmin}),{pruned_class},BS-{bsln_class}".upper().split(",")

    colors_bars = colors[len(lr_data_unique): len(lr_data_unique)+ len(values)]
    for ii, (a_val, a_label) in enumerate(zip(values, labels)):
        bars_2 = ax.bar(x+ii*width, a_val, width=width, label = a_label, color=colors_bars[ii])
        pass
    ax.set_xticklabels(all_xticks)
    xticks = ax.get_xticks()

    x = bars.patches[-1].get_x()
    pos = len(xticks) + width * len(values) / 2
    ax.set_xticks(list(xticks) + [pos])
    all_xticks = list(ax.get_xticklabels())
    all_xticks[-1].set_text("Ref.")
    for ii, a_tick in enumerate(all_xticks):
        a_text = a_tick.get_text()
        if len(a_text) > 5:
            a_tick.set_text(a_text[0:5])
            all_xticks[ii] = a_tick
            pass
    ax.set_xticklabels(all_xticks)
    plt.legend()
    if save_fig:
        dir_name = meta_plot_dict["dir_name"]
        try: os.makedirs(dir_name)
        except: pass
        file_name = meta_plot_dict["file_name"]
        full_name = os.path.join(dir_name, file_name)
        plt.savefig(f"{full_name}")
        pass
    if show_fig:
        plt.show()
        pass
    
    pass
