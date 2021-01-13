from utils.libs.std_python_libs import *
from utils.libs.data_science_libs import *
from utils.libs.graphics_and_interactive_libs import *

def compute_graph_image_psnr_CR(data_tuples, x_axes, y_axes, subject, colors = sns.color_palette()):   
    # Prepare pairs of attributes to be represented
    # one against the other via scatter plot.
    # x_axes = "bpp;file_size_bits".split(";")
    # y_axes = "psnr;CR".split(";")

    pairs_axes = list(itertools.product(x_axes, y_axes))

    # Settle figure grid.
    axes_list = None
    fig, axes = plt.subplots(len(x_axes), len(y_axes), figsize=(20, 10))
    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
        pass

    # Compute graph.
    for ii, (ax, pair_axes) in enumerate(zip(axes_list, pairs_axes)):
        # Prepare data.
        x_axis, y_axis = pair_axes[0], pair_axes[1]
        x = np.array(list(map(lambda item: getattr(item, f"{x_axis}"), data_tuples)))
        y = np.array(list(map(lambda item: getattr(item, f"{y_axis}"), data_tuples)))
        # Create Chart.
        ax.scatter(x, y, marker = 'x', color = colors[ii], label = f'{subject} - {y_axis}')
        # ax.set_xscale('symlog')
        # ax.set_yscale('symlog')
        ax.set_ylabel(f'{y_axis}')
        ax.set_xlabel(f'{x_axis}')
        ax.legend()
        ax.set_title(f'{y_axis.upper()} vs. {x_axis.upper()}')
        pass
    return fig, axes

def graphics_bars_pointplot(dataframe, y_axes, x_axis, grid_shape, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    fig, axes = plt.subplots(*grid_shape, figsize=figsize)
    fig.suptitle(f'{title}', fontsize=15)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pos = 0
        try:
            axes_list = functools.reduce(operator.iconcat, axes, [])
        except:
            axes_list = axes

        _ = graphics_scatterplot(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)
        """
        for ax in axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)]:
            ax.get_xaxis().set_visible(False)
        """
        pos += 1

        _ = graphics_bars_mean_std(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)
    
        for ax in axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)]:
            ax.get_xaxis().set_visible(False)
        pos += 1
    
        _ = graphics_pointplot_mean_std(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)

        for ax in axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)]:
            ax.get_xaxis().set_visible(False)
        pos += 1
    
        _ = graphics_regplot_mean_std(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)
    
        for ax in axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)]:
            ax.get_xaxis().set_visible(False)
        pos += 1

        _ = graphics_boxplot(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)


        for ax in axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)]:
            ax.get_xaxis().set_visible(False)
        pos += 1
    
        _ = graphics_violinplot(
            dataframe = dataframe,
            y_axes = y_axes,
            axes = axes_list[len(y_axes) * pos:len(y_axes) * (pos+1)],
            x_axis = x_axis)
        pass
    return fig, axes

def graphics_scatterplot(dataframe, y_axes, x_axis, grid_shape = None, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ii, (ax, y_axis) in enumerate(zip(axes_list, y_axes)):
        # _ = sns.regplot(x=f"{x_axis}", y=(f"{y_axis}"), data=dataframe, order=1, ax = ax, marker = 'x', color = 'black', label = 'poly order 1°')
        # _ = sns.regplot(x=f"{x_axis}", y=(f"{y_axis}"), data=dataframe, order=2, ax = ax, marker = 'x', color = 'black', label = 'poly order 2°')

        _ = sns.scatterplot(x=f"{x_axis}", y=(f"{y_axis}"), data=dataframe, ax = ax, marker = 'x', color = sns.color_palette()[ii])
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()}', fontsize=10)
        # ax.set_xticklabels(data_xtick_arr, rotation=45)
        # ax.set_xticklabels(data_xtick_arr, rotation=45)
        ax.set_xscale('log')
        pass

    # plt.tight_layout()
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes
    
def graphics_violinplot(dataframe, y_axes, x_axis, grid_shape = None, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        _ = sns.violinplot(x=f"{x_axis}", y=(f"{y_axis}"), data=dataframe, ax = ax, palette="Set3", bw=.2, cut=1, linewidth=1)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()}', fontsize=10)
        ax.set_xticklabels(data_xtick_arr, rotation=45)
        # ax.set_yscale('log')
        pass

    # plt.tight_layout()
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def graphics_boxplot(dataframe, y_axes, x_axis, grid_shape = None, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        _ = sns.boxplot(x=f"{x_axis}", y=(f"{y_axis}"),
            data=dataframe,
            palette=palette, ax = ax)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()}', fontsize=10)
        ax.set_xticklabels(data_xtick_arr, rotation=45)
        # ax.set_yscale('log')
        pass

    # plt.tight_layout()
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def graphics_bars_mean_std(dataframe, y_axes, x_axis, grid_shape = None, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        _ = sns.barplot(x=f"{x_axis}", y=(f"{y_axis}"),
            data=dataframe,
            palette=palette,
            capsize=.0, ax = ax)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()} (mean+std)', fontsize=10)
        ax.set_xticklabels(data_xtick_arr, rotation=45)

    # plt.tight_layout()
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def graphics_bars_mean_std(dataframe, y_axes, x_axis, grid_shape = None, palette="Blues_d", axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        _ = sns.barplot(x=f"{x_axis}", y=(f"{y_axis}"),
            data=dataframe,
            palette=palette,
            capsize=.0, ax = ax)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()} (mean+std)', fontsize=10)
        ax.set_xticklabels(data_xtick_arr, rotation=45)

    # plt.tight_layout()
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def graphics_pointplot_mean_std(dataframe, y_axes, x_axis, grid_shape = None, palette=None, axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        
        _ = sns.pointplot(x=f"{x_axis}", y=(f"{y_axis}"),
            data=dataframe,
            palette=palette,
            capsize=.0, ax = ax)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()} (mean+std)', fontsize=10)
        # ax.set_xticklabels(data_xtick_arr, rotation=45)
        ax.set_xscale('log')
        pass
    
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def graphics_regplot_mean_std(dataframe, y_axes, x_axis, grid_shape = None, palette=None, axes = None, figsize = (15, 5), show_fig = False, title = 'Complex Plot'):
    flag = False
    fig = None
    if axes is None:
        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        fig.suptitle(f'{title}', fontsize=15)
        flag = True
        pass

    data_xtick_arr = \
        np.array(
            np.unique(dataframe[f"{x_axis}"].values),
            dtype=np.int
    )

    try:
        axes_list = functools.reduce(operator.iconcat, axes, [])
    except:
        axes_list = axes
    for ax, y_axis in zip(axes_list, y_axes):
        """
        _ = sns.regplot(x=f"{x_axis}", y=(f"{y_axis}"),
            data=dataframe, color = 'red', label = 'y_axis.upper()', ax = ax)
        """
        _ = sns.regplot(x=f"{x_axis}", y=(f"{y_axis}"), data=dataframe,
                label = f'{y_axis.upper()}',
                # scatter_kws={"s": 80},
                x_estimator=np.mean,
                ax = ax,
                order=4, ci=68)
        # axes[0].get_yaxis().set_visible(False)
        ax.set_title(f'{y_axis.upper()} | poly-regression order 4°', fontsize=10)
        # ax.set_xticklabels(data_xtick_arr, rotation=45)

    
    if flag is False:
        return axes
    else:
        # plt.tight_layout()
        if show_fig: plt.show()
        return fig, axes

def compare_compressions(compression_1_df, compression_2_df):
    # Compare PSNR values between JPEG and Siren, by means of scatterplot.
    fig = plt.figure()

    # Siren results
    plt.scatter(x = compression_1_df['bpp'].values,
        y = compression_1_df['psnr'].values,
        marker = 'x',
        color = sns.color_palette()[0],
        label = 'siren')

    # Jpeg results
    x = np.array(compression_2_df['bpp'].values) #np.array(list(map(lambda item: getattr(item, "bpp"), result_tuples)))
    y = np.array(compression_2_df['psnr'].values) # np.array(list(map(lambda item: getattr(item, "psnr"), result_tuples)))
    plt.scatter(x, y, marker = 'x', color = sns.color_palette()[1], label = 'jpeg')

    # plt.xscale('log')
    plt.ylabel('PSNR')
    plt.xlabel('# bpp')
    plt.legend()
    plt.title('PSNR vs. # Bits')
    plt.show()
    pass