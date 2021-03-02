from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *


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


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation = 45)


def create_pie_plus_hist_charts(
    a_df: pd.DataFrame, group_attributes: list = "quant_techs".split(","),
    target_attr: str = 'quant_techs', plot_confs: dict = None, show_flag: bool = True, axes = []):
    """Comment it.
    Returns:
    --------
    `fig` - if empty dataset, i.e., a_df.shape.[0] == 0, fig will be None\n
    `ax` - if empty dataset, i.e., a_df.shape.[0] == 0, ax will be None\n
    """

    if a_df.shape[0] == 0: return None, None

    a_tmp_df = copy.deepcopy(a_df)
    a_tmp_df[f"{target_attr}"] = list(map(remap_quant_techs, a_tmp_df[f"{target_attr}"].values))
    dfs_groups = a_tmp_df.groupby(by = group_attributes)

    if len(group_attributes) == 1:
        group_attributes = group_attributes[0]

    flag_grid: bool = True
    if len(axes) == 0:
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        fig.suptitle(f"Pie & Bar Charts for {str(group_attributes)}", fontweight="bold")
        flag_grid = False
        pass

    # Pie Chart
    # ------------------------------------------------------------------------------- #
    ax=axes[0]
    dfs_groups.size()\
        .to_frame(name='counts')\
        .plot.pie(y='counts', autopct='%1.1f%%',ax=ax)
    ax.set_title(f"Freq in % of {str(group_attributes)}", fontweight = 'bold')
    ax.set_xlabel(f"{str(group_attributes)}")
    ax.set_ylabel(f"Freq [%]")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    # Bar Chart
    # ------------------------------------------------------------------------------- #
    ax=axes[1]
    bars = dfs_groups.size()\
        .to_frame(name='counts')\
        .plot.bar(y='counts', ax=ax, rot=40)
    rects = bars.patches
    autolabel(rects, ax)
    ax.set_title(f"Counts of {str(group_attributes)}", fontweight = 'bold')
    ax.set_xlabel(f"{str(group_attributes)}")
    ax.set_ylabel(f"Counts")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    # Check whether to save and show resulting plot
    # ------------------------------------------------------------------------------- #
    if flag_grid is False:
        if plot_confs:
            if plot_confs["dir_name"]:
                try:
                    dir_name = plot_confs["dir_name"]
                    try: os.makedirs(dir_name)
                    except: pass
                    file_name = plot_confs["file_name"]
                    full_path = os.path.join(dir_name, file_name)
                    plt.savefig(full_path)
                except: pass
                pass
            pass
        if show_flag:
            plt.show()
        return fig, ax
    
    return None, None
