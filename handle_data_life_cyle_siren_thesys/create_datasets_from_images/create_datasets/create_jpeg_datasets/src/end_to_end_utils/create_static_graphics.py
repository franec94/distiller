from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

import matplotlib


def show_scatterplot_by_hue(a_df: pd.DataFrame, attribute_1: str, attribute_2:str, hue: str, ax)  -> None :
    """TODO COMMENT IT."""
    colors = sns.color_palette()
    # colors = sns.color_palette("Set2")
    title = f"{attribute_1} vs {attribute_2}" # f"{attribute}"

    marker = "o"
    for ii, (a_group, data) in enumerate(a_df.groupby(by = [f"{hue}"])):
        color = colors[ii]
        if a_group == "SIREN":
            # color = colors[0]
            color = colors = sns.color_palette("pastel")[0]
            marker = '*'
            ax.scatter(data[f"{attribute_1}"].values, data[f"{attribute_2}"].values, label=f"{title}: {a_group}",
                    marker = marker, color=color,)
        elif a_group == "BASELINE":
            marker = 's'
            color = colors[3]
            size = 100
            ax.scatter(data[f"{attribute_1}"].values, data[f"{attribute_2}"].values, label=f"{title}: {a_group}",
                    marker = marker, color=color, s=size, edgecolors='b')
        else:
            ax.scatter(data[f"{attribute_1}"].values, data[f"{attribute_2}"].values, label=f"{title}: {a_group}",
                    marker = marker, color=color,)
        pass
    ax.set_title(title)
    ax.legend()
    pass

# =================================================================== #
# Static Scatter Graphics (1)
# =================================================================== #

def create_static_scatter_graphic(a_df: pd.DataFrame, attributes: list, x: str, hue:str = None, **kwargs,) -> object:
    """TODO COMMENT IT."""

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (10, 5)
    if "gridshape" not in kwargs.keys():
        kwargs["gridshape"] = (1, len(attributes))
        pass
    if "title" not in kwargs.keys():
        kwargs["title"] = "Scatter plot"
        pass
    fig, axes = plt.subplots(kwargs["gridshape"][0],
        kwargs["gridshape"][1],
        figsize=kwargs["figsize"])
    plt.grid(True)
    fig.suptitle(kwargs["title"])
    try:
        axes = np.array(list(itertools.chain.from_iterable(axes)))
    except:
        pass
    
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    
    colors = sns.color_palette()
    for ii, (ax, attr) in enumerate(zip(axes, attributes)):
        attribute_1 = f"{x}"
        attribute_2 = f"{attr}"
        
        if hue is None:
            title = f"{attribute_1} vs {attribute_2}" # f"{attribute}"
            ax.scatter(a_df[f"{attribute_1}"].values, a_df[f"{attribute_2}"].values, label=f"{title}",
                    color=colors[ii])
            ax.set_title(title)
            ax.grid(True)
        else:
            show_scatterplot_by_hue(a_df, attribute_1, attribute_2, hue, ax)
            ax.grid(True)
            pass

    # plt.show()
    return fig

# =================================================================== #
# Functions for adding Infos  to a scatter plot
# =================================================================== #

def add_data_point_2_graphics(x:float, y:float, title:str, ax, **kwargs) -> None:
    """TODO comment it."""
    if "color" not in kwargs.keys():
        kwargs["color"] = "red"
    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = "-."

    ax.hlines(y=y, xmin=0, xmax=x, color=kwargs["color"], linestyle=kwargs["linestyle"])
    ax.vlines(x=x, ymin=0, ymax=y, color=kwargs["color"], linestyle=kwargs["linestyle"])
    if title:
        ax.text(x, y, title,
            horizontalalignment='left', # 'center' or 'left' or 'rigth'\
            verticalalignment='center',)
    pass


def add_psnr_data_point(a_df: pd.DataFrame, attribute: str, x: str, hue:str = None, meta_data_boundaries: dict = None, ax= None) -> None:
    """TODO comment it."""
    
    x, y, title = 0.0, 0.0, ""
    jpeg_min_psnr, jpeg_max_psnr = \
        meta_data_boundaries['jpeg_min_psnr'], meta_data_boundaries['jpeg_max_psnr']
    image_bpp = meta_data_boundaries['image_bpp']

    min_siren_row = a_df[ \
        (a_df[f"{hue}"] == "SIREN") & (a_df[f"{attribute}"] <= jpeg_min_psnr) \
        ]\
        .sort_values(by=["psnr"], ascending=False).iloc[0, :]
    siren_mi_row_psnr = min_siren_row["psnr"]
    siren_mi_row_quality = min_siren_row["cmprss-class"]
    siren_mi_row_bpp = min_siren_row["bpp"]
    x, y = siren_mi_row_bpp, siren_mi_row_psnr + 1.0
    # label = f"min siren(psnr|qual.|bpp): ({siren_mi_row_psnr:.2f}|{siren_mi_row_quality}|{siren_mi_row_bpp:.2f})"
    keys = "psnr|qual.|bpp".split("|")
    vals = f"{siren_mi_row_psnr:.2f}|{siren_mi_row_quality}|{siren_mi_row_bpp:.2f}".split("|")
    label = '\n'.join([ f"{k}={v}" for k,v in dict(zip(keys, vals)).items()])
    add_data_point_2_graphics(x=x, y=y, title=label, ax=ax)


    max_siren_row = a_df[ \
        (a_df[f"{hue}"] == "SIREN") & (a_df[f"{attribute}"] <= jpeg_max_psnr) \
        ]\
        .sort_values(by=["psnr"], ascending=False).iloc[0, :]
    siren_ma_row_psnr = max_siren_row["psnr"]
    siren_ma_row_quality = max_siren_row["cmprss-class"]
    siren_ma_row_bpp = max_siren_row["bpp"]
    x, y = siren_ma_row_bpp, siren_ma_row_psnr + 1.0
    keys = "psnr|qual.|bpp".split("|")
    vals = f"{siren_ma_row_psnr:.2f}|{siren_ma_row_quality}|{siren_ma_row_bpp:.2f}".split("|")
    label = '\n'.join([ f"{k}={v}" for k,v in dict(zip(keys, vals)).items()])
    # label = f"min siren(psnr|qual.|bpp): ({siren_ma_row_psnr:.2f}|{siren_ma_row_quality}|{siren_ma_row_bpp:.2f})"
    add_data_point_2_graphics(x=x, y=y, title=label, ax=ax)
    pass


def add_pnsr_boundaries(a_df: pd.DataFrame, attribute: str, x: str, hue:str = None, meta_data_boundaries: dict = None, ax= None) -> None:
    """TODO comment it."""
    jpeg_min_psnr, jpeg_max_psnr = \
        meta_data_boundaries['jpeg_min_psnr'], meta_data_boundaries['jpeg_max_psnr']
    image_bpp = meta_data_boundaries['image_bpp']

    xmax = max(a_df[f"{x}"].values)

    min_jpeg_row = a_df[ \
        (a_df[f"{hue}"] == "JPEG") & (a_df[f"{attribute}"] <= jpeg_min_psnr) \
        ]\
        .sort_values(by=["psnr"], ascending=False).iloc[0, :]
    # pprint(min_jpeg_row)

    assert min_jpeg_row["psnr"] <= jpeg_min_psnr, "min_jpeg_row['psnr'] > jpeg_min_psnr {} > {}".format(min_jpeg_row["psnr"], jpeg_min_psnr)

    jpeg_mi_row_psnr = min_jpeg_row["psnr"]
    jpeg_mi_row_quality = min_jpeg_row["cmprss-class"]
    jpeg_mi_row_bpp = min_jpeg_row["bpp"]
    label = f"min jpeg(psnr|qual.|bpp): ({jpeg_mi_row_psnr:.2f}|{jpeg_mi_row_quality}|{jpeg_mi_row_bpp:.2f})"
    ax.hlines(y=jpeg_min_psnr, xmin=0, xmax=xmax, color='green', linestyle='--', label=label,)

    max_jpeg_row = a_df[ \
        (a_df[f"{hue}"] == "JPEG") & (a_df[f"{attribute}"] <= jpeg_max_psnr) \
        ]\
        .sort_values(by=["psnr"], ascending=False).iloc[0, :]
    # pprint(max_jpeg_row)

    assert max_jpeg_row["psnr"] <= jpeg_max_psnr, "max_jpeg_row['psnr'] > jpeg_max_psnr {} > {}".format(max_jpeg_row["psnr"], jpeg_max_psnr)

    jpeg_ma_row_psnr = max_jpeg_row["psnr"]
    jpeg_ma_row_quality = max_jpeg_row["cmprss-class"]
    jpeg_ma_row_bpp = max_jpeg_row["bpp"]
    label = f"max jpeg(psnr|qual.|bpp): ({jpeg_ma_row_psnr:.2f}|{jpeg_ma_row_quality}|{jpeg_ma_row_bpp:.2f})"
    ax.hlines(y=jpeg_max_psnr, xmin=0, xmax=xmax, color='green', linestyle='--', label=label,)

    label = f"Image Bpp: {image_bpp:.2f}"
    ymax = max(a_df[f"{attribute}"].values)
    ax.vlines(x=image_bpp, ymin=0, ymax=ymax, color='black', linestyle='--', label=label,)
    pass

# =================================================================== #
# Static Scatter Graphics (2)
# =================================================================== #

def create_static_scatter_graphic_with_boundaries(a_df: pd.DataFrame, attributes: list, x: str, hue:str = None, meta_data_boundaries: dict = None, **kwargs,) -> object:
    """TODO COMMENT IT."""
    
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (10, 5)
    if "gridshape" not in kwargs.keys():
        kwargs["gridshape"] = (1, len(attributes))
        pass
    if "title" not in kwargs.keys():
        kwargs["title"] = "Scatter plot"
        pass

    fig, axes = plt.subplots(kwargs["gridshape"][0],
        kwargs["gridshape"][1],
        figsize=kwargs["figsize"])
    plt.grid(True)
    fig.suptitle(kwargs["title"])
    try:
        axes = np.array(list(itertools.chain.from_iterable(axes)))
    except:
        pass
    
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    
    colors = sns.color_palette()
    for ii, (ax, attr) in enumerate(zip(axes, attributes)):
        attribute_1 = f"{x}"
        attribute_2 = f"{attr}"
        ax.grid(True)
        
        if hue is None:
            if attribute_2 == "psnr" and meta_data_boundaries is not None:
                add_pnsr_boundaries(a_df, attribute_2, x, hue, meta_data_boundaries, ax)
                add_psnr_data_point(a_df, attribute_2, x, hue, meta_data_boundaries, ax)
            title = f"{attribute_1} vs {attribute_2}" # f"{attribute}"
            ax.scatter(a_df[f"{attribute_1}"].values, a_df[f"{attribute_2}"].values, label=f"{title}",
                    color=colors[ii], edgecolors='b')
            ax.set_title(title)
        else:
            if attribute_2 == "psnr" and meta_data_boundaries is not None:
                add_pnsr_boundaries(a_df, attribute_2, x, hue, meta_data_boundaries, ax)
                add_psnr_data_point(a_df, attribute_2, x, hue, meta_data_boundaries, ax)
            show_scatterplot_by_hue(a_df, attribute_1, attribute_2, hue, ax)
            pass

    # plt.show()
    return fig
