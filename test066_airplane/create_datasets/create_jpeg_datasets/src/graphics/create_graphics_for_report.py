from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *


def add_custom_legend(ax) -> None:
    """Add custom legend."""
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1, title='Quant techs')
    pass

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


def show_pie_chart(
    a_df: pd.DataFrame,
    group_attributes:list, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """TODO comment it."""

    if len(group_attributes) == 1:
        a_clss = group_attributes[0]
        clss_list = list(a_df[f"{a_clss}"].unique())
        colors = sns.color_palette()
        n = len(clss_list)
        palette = dict(zip(clss_list, colors[0:n]))
        pass

    flag_not_ax: bool = False
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        flag_not_ax = True
        pass
    
    dfs_groups = a_df.groupby(by = group_attributes)
    if len(group_attributes) == 1:
        tmp_df = dfs_groups.size().to_frame(name='counts')
        colors = [palette[x] for x in tmp_df.index]
        tmp_df.plot.pie(y='counts', autopct='%1.1f%%',ax=ax, colors=colors)
    else:
        dfs_groups.size().to_frame(name='counts').plot.pie(y='counts', autopct='%1.1f%%',ax=ax)
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass

    add_custom_legend(ax=ax)

    if flag_not_ax and save_fig:
        try: os.makedirs(os.path.dirname(figname))
        except Exception as err:
            print(f"{str(err)}")
            pass
        plt.savefig(figname)
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


def show_bar_plot(
    a_df: pd.DataFrame,
    x:str=None, y:str="psnr", hue:str=None, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """Create Image with boxplots that show within them Instances numbers in each class."""
    
    flag_not_ax: bool = False
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        flag_not_ax = True
        pass
    ax_bars = sns.barplot(x=f"{x}", y=f"{y}", data=a_df, ax=ax)
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass

    add_custom_legend(ax=ax)

    if flag_not_ax and save_fig:
        try:
            plt.savefig(figname)
        except Exception as err:
            print(f"{str(err)}")
            pass
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


def show_bar_plot_2(
    a_df: pd.DataFrame,
    x:str=None, y:str="psnr", hue:str=None, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """Create Image with boxplots that show within them Instances numbers in each class."""
    
    flag_not_ax: bool = False
    if not ax:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize=(7,5)
        fig, ax = plt.subplots(1,1,figsize=figsize)
        flag_not_ax = True
        pass

    # gpby_cols = "quant_techs,nbits".split(",")
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
    ax_bar = tmp_df.plot.bar(rot=40, ax=ax)
    # ax_bar = tmp_df.plot.pie(autopct='%1.1f%%',ax=ax, subplots=True)
    autolabel(rects = ax.patches, ax=ax_bar)
    title = kwargs["ylabel"]["ylabel"]
    ax.legend(title=f'{title}')

    if "xlabel" in kwargs.keys():
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass
    add_custom_legend(ax=ax)

    if flag_not_ax and save_fig:
        try: os.makedirs(os.path.dirname(figname))
        except Exception as err:
            print(f"{str(err)}")
            pass
        plt.savefig(figname)
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


def show_scatter_plot(
    a_df: pd.DataFrame,
    x:str=None, y:str="psnr", hue:str=None, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """Create Image with boxplots that show within them Instances numbers in each class."""
    
    flag_not_ax: bool = False
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        flag_not_ax = True
        pass
    ax_violinplot = sns.scatterplot(x=f"{x}", y=f"{y}", data=a_df, ax=ax, hue=f"{hue}")
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass
    add_custom_legend(ax=ax)
    if flag_not_ax and save_fig:
        try: os.makedirs(os.path.dirname(figname))
        except Exception as err:
            print(f"{str(err)}")
            pass
        print(f"Saving file: {figname}...")
        plt.savefig(figname)
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


def show_violin_and_instances_no(
    a_df: pd.DataFrame,
    x:str=None, y:str="psnr", hue:str=None, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """Create Image with boxplots that show within them Instances numbers in each class."""

    clss_list = list(a_df[f"{hue}"].unique())
    colors = sns.color_palette()
    n = len(clss_list)
    palette = dict(zip(clss_list, colors[0:n]))
    
    flag_not_ax: bool = False
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        flag_not_ax = True
        pass
    ax_violinplot = sns.violinplot(x=f"{x}", y=f"{y}", data=a_df, ax=ax, palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    if show_jitter:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = sns.swarmplot(x=f"{x}", y=f"{y}", data=a_df, color="grey",ax=ax)
            pass
        pass
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass

    # Calculate number of obs per group & median to position labels
    medians = a_df.groupby([f'{hue}'])[f'{y}'].median()
    nobs = a_df[f'{hue}'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    counts = a_df[f'{hue}'].value_counts()
    for tick,label in zip(pos,ax.get_xticklabels()):
        a_clss = label.get_text()
        a_nobs = "n: {}".format(int(counts[f"{a_clss}"]))
        ax_violinplot.text(tick, medians[f"{a_clss}"] + 0.03, a_nobs,
            horizontalalignment='center', size='x-small',
            color='black', weight='semibold')
        pass

    add_custom_legend(ax=ax)

    if flag_not_ax and save_fig:
        try:
            plt.savefig(figname)
        except Exception as err:
            print(f"{str(err)}")
            pass
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


def show_boxplot_and_instances_no(
    a_df: pd.DataFrame,
    x:str=None, y:str="psnr", hue:str=None, ax = None,
    save_fig: bool = False, show_fig: bool = False,
    figname:str="boxplot_with_instances_no.png", show_jitter: bool = False, **kwargs) -> object:
    """Create Image with boxplots that show within them Instances numbers in each class."""

    clss_list = list(a_df[f"{hue}"].unique())
    colors = sns.color_palette()
    n = len(clss_list)
    palette = dict(zip(clss_list, colors[0:n]))
    
    flag_not_ax: bool = False
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        flag_not_ax = True
        pass
    ax_boxplot = sns.boxplot(x=f"{x}", y=f"{y}", data=a_df, ax=ax, palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    if show_jitter:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = sns.swarmplot(x=f"{x}", y=f"{y}", data=a_df, color="grey", ax=ax)
            pass
        pass
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(**kwargs["ylabel"])
    if "title" in kwargs.keys():
        ax.set_title(**kwargs["title"])
        pass

    # Calculate number of obs per group & median to position labels
    medians = a_df.groupby([f'{hue}'])[f'{y}'].median()
    nobs = a_df[f'{hue}'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    counts = a_df[f'{hue}'].value_counts()
    for tick,label in zip(pos,ax.get_xticklabels()):
        a_clss = label.get_text()
        a_nobs = "n: {}".format(int(counts[f"{a_clss}"]))
        # ax_boxplot.text(pos[tick], medians[tick] + 0.03, a_nobs,
        ax_boxplot.text(tick, medians[f"{a_clss}"] + 0.03, a_nobs,
            horizontalalignment='center', size='x-small',
            color='black', weight='semibold')
        pass
    
    add_custom_legend(ax=ax)

    if flag_not_ax and save_fig:
        try:
            plt.savefig(figname)
        except Exception as err:
            print(f"{str(err)}")
            pass
    if flag_not_ax and show_fig:
        plt.show()
    if flag_not_ax:
        return fig, ax
    return ax


# A Box Plots
# ----------------------------------------------------------------------------------------------------- #
def create_a_box_plot(
    a_df: pd.DataFrame,
    figures_list: list,
    x, y, hue, ax = None, save_fig: bool = True,
    show_fig: bool = False, show_jitter: bool = False,
    dir_name: str = "."):
    """TODO comment it."""
    # x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    if y == "psnr":
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel="psnr [db]".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"Psnr [db] by {x}", fontweight="bold"
            )
        )
    else:
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel=f"{y}".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"{y.upper()} by {x}", fontweight="bold"
            )
        )
    
    if ax is None:
        file_name = f"{x}_vs_{y}_boxplot.png"
        dir_name_boxplots = os.path.join(dir_name, "boxplot_charts")
        full_name = os.path.join(dir_name_boxplots, file_name)
        fig, ax = show_boxplot_and_instances_no(
            a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
            show_fig = show_fig, show_jitter = show_jitter,
            figname=full_name, save_fig = save_fig, **kwargs
        )
        figures_list.append(full_name)
        return fig, ax
    file_name = f"{x}_vs_{y}_boxplot.png"
    dir_name_boxplots = os.path.join(dir_name, "boxplot_charts")
    full_name = os.path.join(dir_name_boxplots, file_name)
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
        show_fig = show_fig, show_jitter = show_jitter,
        figname=full_name, save_fig = save_fig, **kwargs
    )
    figures_list.append(full_name)
    return ax



# Summary Box Plots
# ----------------------------------------------------------------------------------------------------- #
def create_summary_box(a_df: pd.DataFrame, figures_list: list, dir_name: str):
    """TODO comment it."""
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: boxplot for psnr,bpp,ssim, and CR", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, **kwargs
    )

    ax = axes[1]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, **kwargs
    )

    ax = axes[2]
    x, y, hue = "cmprss-class-2,ssim,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, **kwargs
    )

    ax = axes[3]
    x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, **kwargs
    )

    file_name = f"summary_boxplot.png"
    dir_name_boxplots = os.path.join(dir_name, "boxplot_charts")
    full_name = os.path.join(dir_name_boxplots, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass


# A Violin Plots
# ----------------------------------------------------------------------------------------------------- #
def create_a_violin_plot(
    a_df: pd.DataFrame,
    figures_list: list,
    x, y, hue, 
    dir_name: str = ".",
    ax = None, show_fig: bool = False,
    show_jitter:bool = False, save_fig:bool=False):
    """TODO comment it."""
    # x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    if y == "psnr":
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel="psnr [db]".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"Psnr [db] by {x}", fontweight="bold"
            )
        )
    else:
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel=f"{y}".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"{y.upper()} by {x}", fontweight="bold"
            )
        )
    if ax is None:
        file_name = f"{x}_vs_{y}_violinplot.png"
        dir_name_violins = os.path.join(dir_name, "violin_charts")
        full_name = os.path.join(dir_name_violins, file_name)
        fig, ax = show_violin_and_instances_no(
            a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
            show_fig = show_fig, show_jitter = show_jitter, ax=ax,
            save_fig = save_fig, figname=full_name, **kwargs
        )
        figures_list.append(full_name)
        return fig,ax
    file_name = f"{x}_vs_{y}_violinplot.png"
    dir_name_violins = os.path.join(dir_name, "violin_charts")
    full_name = os.path.join(dir_name_violins, file_name)
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = show_fig, show_jitter = show_jitter, ax=ax,
        save_fig = save_fig, figname=full_name, **kwargs
    )
    figures_list.append(full_name)
    return ax


# Summary Violin Plots
# ----------------------------------------------------------------------------------------------------- #
def create_summary_violin(a_df: pd.DataFrame, figures_list: list, dir_name: str):
    """TODO comment it."""
    show_jitter = True
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: violin plots for psnr,bpp,ssim,and CR", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[1]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[2]
    x, y, hue = "cmprss-class-2,ssim,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[3]
    x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
        show_fig = False, show_jitter = False, save_fig = False, **kwargs
    )

    file_name = f"summary_violinplot.png"
    dir_name_violins = os.path.join(dir_name, "violin_charts")
    full_name = os.path.join(dir_name_violins, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass


# A Bar Plots
# ----------------------------------------------------------------------------------------------------- #
def create_a_bar_plot(
    a_df: pd.DataFrame,
    figures_list: list,
    x, y, hue,
    dir_name: str = ".",
    ax = None, show_fig: bool = False,
    show_jitter:bool = False, save_fig:bool=False):
    """TODO comment it."""
    # x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    if y == "psnr":
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel="psnr [db]".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"Psnr [db] by {x}", fontweight="bold"
            )
        )
    else:
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel=f"{y}".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"{y.upper()} by {x}", fontweight="bold"
            )
        )
    if ax is None:
        file_name = f"{x}_vs_{y}_barplot.png"
        dir_name_bar_charts = os.path.join(dir_name, "bar_charts")
        try: os.makedirs(dir_name_bar_charts)
        except: pass
        full_name = os.path.join(dir_name_bar_charts, file_name)
        fig, ax = show_bar_plot(
            a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
            show_fig = show_fig, show_jitter = show_jitter, save_fig = save_fig, figname=full_name, **kwargs
        )
        figures_list.append(full_name)
        return fig, ax
    file_name = f"{x}_vs_{y}_barplot.png"
    dir_name_bar_charts = os.path.join(dir_name, "bar_charts")
    full_name = os.path.join(dir_name_bar_charts, file_name)
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_jitter = show_jitter, ax=ax,
        save_fig = save_fig, figname=full_name, **kwargs
    )
    figures_list.append(full_name)
    pass


def create_a_bar_plot_counts_ws_index(
    a_df: pd.DataFrame,
    figures_list: list,
    index:str,
    y:str, hue:str, ax = None,
    dir_name: str = ".", save_fig: bool = True):
    """TODO comment it."""
    # x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    if y == "psnr":
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel="psnr [db]".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"Psnr [db] by {x}", fontweight="bold"
            )
        )
    else:
        kwargs = dict(
            xlabel=dict(
                xlabel=f"{index}".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel=f"Count".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"{y.upper()} by {index}", fontweight="bold"
            )
        )
    file_name = f"{y}_vs_{index}_count_barplot.png"
    # dir_name_bar_charts = os.path.join(dir_name, "bar_charts")
    dir_name_bar_charts = dir_name
    try: os.makedirs(dir_name_bar_charts)
    except: pass
    full_name = os.path.join(dir_name_bar_charts, file_name)
    if ax:
        ax = show_bar_plot_2(
            a_df = a_df, x = f"{index}", y = f"{y}", hue = f"{hue}", ax=ax,
            show_fig = False, show_jitter = True, save_fig = save_fig, figname=full_name, **kwargs
        )
        return ax
    else:
        fig, ax = show_bar_plot_2(
            a_df = a_df, x = f"{index}", y = f"{y}", hue = f"{hue}", ax=ax,
            show_fig = False, show_jitter = True, save_fig = save_fig, figname=full_name, **kwargs
        )
        figures_list.append(full_name)
        return fig, ax

# Summary Bar Plots
# ----------------------------------------------------------------------------------------------------- #
def create_summary_barplots(a_df: pd.DataFrame, figures_list: list, dir_name: str):
    """TODO comment it."""
    show_jitter = True
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: violin plots for psnr,bpp,ssim,and CR", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_bar_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[1]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_bar_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[2]
    x, y, hue = "cmprss-class-2,ssim,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_bar_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[3]
    x, y, hue = "cmprss-class-2,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_bar_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
        show_fig = False, show_jitter = False, save_fig = False, **kwargs
    )

    file_name = f"summy_barplot.png"
    dir_name_bar_charts = os.path.join(dir_name, "bar_charts")
    full_name = os.path.join(dir_name_bar_charts, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass


# A Scatter Plots
# ----------------------------------------------------------------------------------------------------- #
def create_a_scatter_plot(
        a_df: pd.DataFrame,
    figures_list: list,
    x, y, hue, ax = None,
    save_fig: bool = True, show_fig:bool = False,
    dir_name: str = ".", file_name: str = None):
    """TODO comment it."""
    # x, y, hue = "bpp,mse,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    if y == "psnr":
        kwargs = dict(
            xlabel=dict(
                xlabel=f"{x}".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel="psnr [db]".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"Psnr [db] by {x}", fontweight="bold"
            )
        )
    else:
        kwargs = dict(
            xlabel=dict(
                xlabel="cmprss-class-2".upper(), fontweight="bold"
            ),
            ylabel=dict(
                ylabel=f"{y}".upper(), fontweight="bold"
            ),
            title=dict(
                label=f"{y.upper()} by {x}", fontweight="bold"
            )
        )

    if ax is None:
        if file_name is None:
            file_name = f"{x}_vs_{y}_scatter.png"
        dir_name_scatter_charts = os.path.join(dir_name, "scatter_charts")
        full_name = os.path.join(dir_name_scatter_charts, file_name)
        fig, ax = show_scatter_plot(
            a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
            show_fig = show_fig, show_jitter = True, save_fig = save_fig, figname=full_name, **kwargs
        )
        figures_list.append(full_name)
        return fig, ax

    if file_name is None:
        file_name = f"{x}_vs_{y}_scatter.png"
    dir_name_scatter_charts = os.path.join(dir_name, "scatter_charts")
    full_name = os.path.join(dir_name_scatter_charts, file_name)
    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
        show_fig = show_fig, show_jitter = True, save_fig = save_fig, figname=full_name, **kwargs
    )
    return ax


# Summary Scatter Plots
# ----------------------------------------------------------------------------------------------------- #
def create_summary_scatter(a_df: pd.DataFrame, figures_list: list, dir_name: str):
    """TODO comment it."""
    show_jitter = True
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: scatter plots for psnr,bpp,ssim,and CR", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "bpp,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[1]
    x, y, hue = "bpp,ssim,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[2]
    x, y, hue = "bpp,CR,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = show_jitter, **kwargs
    )

    ax = axes[3]
    x, y, hue = "bpp,mse,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax = ax,
        show_fig = False, show_jitter = False, save_fig = False, **kwargs
    )

    file_name = f"summy_scatter.png"
    dir_name_scatter_charts = os.path.join(dir_name, "scatter_charts")
    full_name = os.path.join(dir_name_scatter_charts, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass


# Summary Box-Violin Plots
def create_summary_box_violin(a_df: pd.DataFrame, figures_list: list, dir_name: str):
    """TODO comment it."""
    show_jitter = True
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: mix box,violin plots for psnr,bpp", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, show_jitter = False, **kwargs
    )

    ax = axes[1]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = False, **kwargs
    )

    ax = axes[2]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = False, **kwargs
    )

    ax = axes[3]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )

    ax = show_violin_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax=ax,
        show_fig = False, show_jitter = False, save_fig = False, **kwargs
    )

    file_name = f"summary_mix_box_violin_psnr_bpp.png"
    dir_name_mix_charts = os.path.join(dir_name, "mix_charts")
    full_name = os.path.join(dir_name_mix_charts, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass


# Summary BVPS Plots
# ----------------------------------------------------------------------------------------------------- #
def create_summary_mix_bvps(a_df: pd.DataFrame, figures_list: list, dir_name:str):
    """TODO comment it."""
    show_jitter = True
    fig, axes = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("Summary Chart: mix box,violin,pie,scater plots for psnr,bpp", fontweight="bold")
    if axes.shape[0] > 1:
        axes = axes.flatten()
        pass

    ax = axes[0]
    x, y, hue = "cmprss-class-2,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax=ax, show_jitter = False, **kwargs
    )

    ax = axes[1]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel="psnr [db]".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"Psnr [db] by {x}", fontweight="bold"
        )
    )
    ax = show_boxplot_and_instances_no(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = False, **kwargs
    )

    ax = axes[2]
    x, y, hue = "cmprss-class-2,bpp,cmprss-class-2".split(",")
    group_attributes = "cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="classes".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"Freq [%]", fontweight="bold"
        ),
        title=dict(
            label=f"Freq [%] Classes", fontweight="bold"
        )
    )
    ax = show_pie_chart(
        a_df = a_df, group_attributes=group_attributes, hue = f"{hue}",
        show_fig = False, ax = ax, show_jitter = False, **kwargs
    )

    ax = axes[3]
    x, y, hue = "bpp,psnr,cmprss-class-2".split(",")
    # a_df = copy.deepcopy(jpg_siren_df)
    kwargs = dict(
        xlabel=dict(
            xlabel="cmprss-class-2".upper(), fontweight="bold"
        ),
        ylabel=dict(
            ylabel=f"{y}".upper(), fontweight="bold"
        ),
        title=dict(
            label=f"{y.upper()} by {x}", fontweight="bold"
        )
    )

    ax = show_scatter_plot(
        a_df = a_df, x = f"{x}", y = f"{y}", hue = f"{hue}", ax=ax,
        show_fig = False, show_jitter = False, save_fig = False, **kwargs
    )

    file_name = f"summary_mix_charts_psnr_bpp.png"
    dir_name_mix_charts = os.path.join(dir_name, "mix_charts")
    full_name = os.path.join(dir_name_mix_charts, file_name)
    figures_list.append(full_name)
    plt.savefig(full_name)
    pass
