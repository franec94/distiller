import PIL
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *
from src.data_loaders import dataset_loaders

# ======================================== #
# Scatter Plots Datasets
# ======================================== #

def show_jpeg_df(jpeg_df=pd.DataFrame(), ax=None, **kwargs) -> None:
    if jpeg_df.shape[0] == 0 or ax is None: return
    colors = "orange,blue".split(",")
    image_names = "cameramen,test066".split(",")
    markers_list = "x,x,".split(",")
    marker_name_dict = dict(zip(image_names, markers_list))
    colors_name_dict = dict(zip(image_names, colors))
    for ii, (gk, gdata) in enumerate(jpeg_df.groupby(by=["image_name"])):
        if gk not in image_names: continue
        x, y = gdata["bpp"].values, gdata["psnr"].values
        marker = marker_name_dict[gk]
        color = colors_name_dict[gk]
        # plt.scatter(x, y, marker=marker, s=100, label=f"{gk}(Jpeg)", color=color, edgecolors="black")
        plt.scatter(x, y, marker="o", s=100, label=f"{gk}(Jpeg)", color="white", edgecolors="blue")
        pass
    pass


def show_prune_df(prune_df=pd.DataFrame(), ax=None, *kwargs) -> None:
    if prune_df.shape[0] == 0 or ax is None: return

    colors = "orange,blue".split(",")
    image_names = "cameramen,test066".split(",")
    markers_list = "+,+,".split(",")
    marker_name_dict = dict(zip(image_names, markers_list))

    colors_name_dict = dict(zip(image_names, colors))
    for gk, gdata in prune_df.groupby(by=["image_name_2"]):
        x, y = gdata["bpp"].values, gdata["psnr"].values
        kf = None
        for k in image_names:
            if k in gk:
                kf = k
                break
        if not kf: continue
        marker = marker_name_dict[kf]
        color = colors_name_dict[kf]
        x, y = gdata["bpp"].values, gdata["psnr"].values
        # plt.scatter(x, y, marker=marker, s=100, label=gk, color=color, edgecolors="black")
        # plt.scatter(x, y, marker=marker, s=100, label=gk, color="green", edgecolors="black")
        plt.scatter(x, y, marker="o", s=100, label=gk, color="white", edgecolors="green")
        pass
    pass


def show_quant_df(quant_df=pd.DataFrame(), ax=None, *kwargs) -> None:
    if quant_df.shape[0] == 0 or ax is None: return
    for gk, gdata in quant_df.groupby(by=["quant_techs_2", "nbits"]):
        tech, nbits = gk
        if nbits == 5: continue
        gdata = gdata.sort_values(by=["psnr"], ascending=False)
        x, y = gdata["bpp"].values[0:10], gdata["psnr"].values[0:10]
        label = f"{gk[0]}({gk[1]})"
        plt.scatter(x, y, s=100, marker="D", label=label, edgecolors="black")
        pass
    pass


def show_baseline_df(a_df=pd.DataFrame(), thsd_list:list = [], ax=None, *kwargs) -> None:
    if a_df.shape[0] == 0 or ax is None: return
    
    if thsd_list != []:
        pos = a_df["psnr"] <= thsd_list[0]
        tmp_df = a_df[pos]
        x, y = tmp_df["bpp"].values[0:], tmp_df["psnr"].values[0:]
        ax.scatter(
            x, y, label="BL-U.P.", color="white", edgecolors="red", marker="o"
        )

        pos = (a_df["psnr"] >= thsd_list[0]) & (a_df["bpp"] <= thsd_list[1])
        tmp_df = a_df[pos]
        x, y = tmp_df["bpp"].values[0:], tmp_df["psnr"].values[0:]
        ax.scatter(
            x, y, label="BL-M.", color="white", edgecolors="green", marker="o"
        )

        pos = a_df["bpp"] >= thsd_list[1]
        tmp_df = a_df[pos]
        x, y = tmp_df["bpp"].values[0:], tmp_df["psnr"].values[0:]
        ax.scatter(
            # x, y, label="BL-O.P.", color="white", edgecolors="orange", marker="o"
            x, y, label="BL-O.P.", color="white", edgecolors="black", marker="o"
        )
    else:
        x, y = a_df["bpp"].values[0:], a_df["psnr"].values[0:]
        ax.scatter(
            x, y, label="baseline", color="white", edgecolors="black", marker="o"
        )
    pass


# ======================================== #
# Scatter Plots Merged Datasets
# ======================================== #

def show_merged_jpeg_data_points(a_df:pd.DataFrame=pd.DataFrame(), x_attr:str="bpp", y_attr:str="psnr", gk=None, ax=None, payload_jpeg:dict=None, **kwargs) -> None:
    if a_df.shape[0] == 0: return
    if ax is None: return
    if gk is None : return

    if not isinstance(gk, list):
        gk = [gk]

    try:
        for ii, (k, data) in enumerate(a_df.groupby(by = gk)):
            x, y = data[x_attr].values, data[y_attr].values
            label=f"{k}(JPG)"
            if payload_jpeg:
                if k in payload_jpeg.keys():
                    try:
                        k_style = payload_jpeg[k]
                        ax.scatter(x, y, label=label, marker=k_style["marker"], color=k_style["color"], edgecolor=k_style["edgecolor"])
                    except:
                        ax.scatter(x, y, label=label, marker="o")
                        pass
                else: 
                    ax.scatter(x, y, label=label, marker="o")
                    pass
                pass
            else:
                if ii % 2 == 0:
                    ax.scatter(x, y, label=label, marker="+")
                else:
                    ax.scatter(x, y, label=label, marker="+")
                pass
            pass
    except Exception as err:
        print(f"{str(err)}")
        pass
    pass


def show_merged_pruned_data_points(a_df:pd.DataFrame=pd.DataFrame(), x_attr:str="bpp", y_attr:str="psnr", gk=None, ax=None, payload_pruned:dict=None, **kwargs) -> None:
    if a_df.shape[0] == 0: return
    if ax is None: return
    if gk is None : return

    if not isinstance(gk, list):
        gk = [gk]

    try:
        for ii, (k, data) in enumerate(a_df.groupby(by = gk)):
            x, y = data[x_attr].values, data[y_attr].values
            label=f"{k}(AGP)"
            if payload_pruned:
                if k in payload_pruned.keys():
                    try:
                        k_style = payload_pruned[k]
                        ax.scatter(x, y, label=label, marker=k_style["marker"], color=k_style["color"], edgecolor=k_style["edgecolor"], s=100)
                    except:
                        ax.scatter(x, y, label=label, marker="o")
                        pass
                else: 
                    ax.scatter(x, y, label=label, marker="o")
                    pass
                pass
            else:
                if ii % 2 == 0:
                    ax.scatter(x, y, label=label, marker="o", edgecolor="black", s=100)
                else:
                    ax.scatter(x, y, label=label, marker="o", edgecolor="black", s=100)
                pass
            pass
    except Exception as err:
        print(f"{str(err)}")
        pass
    pass


def show_quant_data_points(a_df:pd.DataFrame=pd.DataFrame(), x_attr:str="bpp", y_attr:str="psnr", gk=None, ax=None, payload_quanted:dict=None, **kwargs) -> None:
    if a_df.shape[0] == 0: return
    if ax is None: return
    if gk is None : return

    if not isinstance(gk, list):
        gk = [gk]

    try:
        for ii, (k, data) in enumerate(a_df.groupby(by = gk)):
            tmp_x = data.sort_values(by=[x_attr], ascending=False)[x_attr].values[0:10]
            tmp_y = data.sort_values(by=[y_attr], ascending=False)[y_attr].values[0:10]

            x, y = tmp_x, tmp_y
            label=f"{k}(RLQ)"
            if payload_quanted:
                if k in payload_quanted.keys():
                    try:
                        k_style = payload_quanted[k]
                        ax.scatter(x, y, label=label, marker=k_style["marker"], color=k_style["color"], edgecolor=k_style["edgecolor"])
                    except:
                        ax.scatter(x, y, label=label, marker="o")
                        pass
                else: 
                    ax.scatter(x, y, label=label, marker="o")
                    pass
                pass
            else:
                if k[0] == "cameramen":
                    ax.scatter(x, y, label=label, marker="^", s=100, edgecolor="black")
                else:
                    ax.scatter(x, y, label=label, marker="v", s=100, edgecolor="black")
                pass
            pass
    except Exception as err:
        print(f"{str(err)}")
        pass
    pass


def show_images_within_plot(
    pos_x=6.3, pos_y=37, pos_2_x=4.5, pos_2_y=37,
    delta_x=0.5, delta_2_x=0.5,
    images_list:list = [],
    delta_y=0.5, delta_y_2=0.5, ax=None):
    
    if ax is None: return
    if images_list == []: return

    image_name="cameramen"
    if image_name in images_list:
        image_cameramen = dataset_loaders.load_image_by_name(image_name=f"{image_name}", cropped_center=256)

        image_cameramen_smaller = image_cameramen.resize((160, 160), PIL.Image.ANTIALIAS, )
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(image_cameramen_smaller, cmap='gray',)
                , (pos_x, pos_y)
                , frameon=False
            ) 
        )
        ax.text(pos_x-delta_x, pos_y + delta_y, "cameramen (256x256)", fontsize=12, fontweight="bold")

    image_name="test066"
    if image_name in images_list:
        image_test066 = dataset_loaders.load_image_by_name(image_name=f"{image_name}", cropped_center=256)

        image_test066_smaller = image_test066.resize((160, 160), PIL.Image.ANTIALIAS, )
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(image_test066_smaller, cmap='gray',)
                , (pos_2_x, pos_2_y)
                , frameon=False
            )
        )
        ax.text(pos_2_x-delta_2_x, pos_2_y + delta_y_2, "test066 (256x256)", fontsize=12, fontweight="bold")
    pass

# ======================================== #
# Scatter Plots Fixed Points
# ======================================== #

def show_baseline_fixed_points(a_df, ax, pairs_hf_hl:list = [], pos_alignment: list = [], **kwargs):
    """Add fixed data points to a given input image."""
    if a_df.shape[0] == 0 or not ax: return
    if not ax: return
    if len(pairs_hf_hl) == 0: return

    if "horizontalalignment" not in kwargs.keys():
        kwargs["horizontalalignment"] = "center"
        pass
    if len(pos_alignment) == 0:
        pos_alignment = [kwargs["horizontalalignment"]] * len(pairs_hf_hl)
        pass
    # pick_cols = ["bpp", "psnr"]
    # pick_cols_2 = ["n_hf", "n_hl"]
    for ii, (n_hf, n_hl) in enumerate(pairs_hf_hl):
        pos = (a_df["n_hf"] == n_hf) & (a_df["n_hl"] == n_hl)
        
        bpp, psnr_val = a_df[pos]["bpp"].values, a_df[pos]["psnr"].values
        bpp, psnr_val = bpp[0], psnr_val[0]

        x, y = a_df[pos]["bpp"].values, a_df[pos]["psnr"].values
        x, y = x[0], y[0]

        n_hf, n_hl = a_df[pos]["n_hf"].values, a_df[pos]["n_hl"].values
        n_hf, n_hl = n_hf[0], n_hl[0]
        
        msg = f"n_hf={n_hf}\nn_hl={n_hl}\n({bpp:.2f},{psnr_val:.2f})"
        if ii % 2 == 0:
            ax.text(x=x, y=y-2, s = msg,
                # horizontalalignment=kwargs["horizontalalignment"],
                horizontalalignment=pos_alignment[ii],
                fontdict={'fontsize': 12, 'fontweight': 'bold'})
        else:
            ax.text(x=x, y=y+1.25, s = msg,
                # horizontalalignment=kwargs["horizontalalignment"],
                horizontalalignment=pos_alignment[ii],
                fontdict={'fontsize': 12, 'fontweight': 'bold'})
            pass
        ax.scatter(x=x, y=y, marker="d", color="yellow", edgecolor="black", s=125)
        ax.scatter(x=x, y=y, marker=".", color="red")
        pass
    pass


def add_jpeg_fixed_points(jpeg_df=pd.DataFrame(), image_name=None, qualities=[], ax=None, bpp_flag: bool = False, **kwargs) -> None:
    """Add fixed data points to a given input image."""

    if jpeg_df.shape[0] == 0 or not ax: return

    tmp_jpeg_image_df = jpeg_df[jpeg_df["image_name"] == image_name]
    if tmp_jpeg_image_df.shape[0] == 0: return

    if "horizontalalignment" not in kwargs.keys():
        kwargs["horizontalalignment"] = "center"

    pos = tmp_jpeg_image_df["bpp"] <= 1.0
    a_row = tmp_jpeg_image_df[pos].sort_values(by = ["psnr"], ascending=False).head(1)
    if a_row.shape[0] != 0:
        pnsr = a_row["psnr"].values[0]
        bpp = a_row["bpp"].values[0]
        a_quality = a_row["quality"].values[0]

        x, y = bpp, pnsr
        if bpp_flag:
            ax.scatter(x, y, marker='D', color="yellow", s=150, edgecolors='black')
            ax.scatter(x, y, marker='.', color="red", edgecolors='black')
            # msg: str = f"JPG:{a_quality}%->({x:.2f},{y:.2f})"
            msg: str = f"JPG:{a_quality}%\n({x:.2f},{y:.2f})"
            ax.text(x=x, y=y, s = msg,
                horizontalalignment=kwargs["horizontalalignment"],
                fontdict={'fontsize': 14, 'fontweight': 'bold'})
            pass
        xmin, xmax = ax.get_xlim()
        ax.hlines(y = y, xmin=xmin, xmax=xmax, linestyle="--",  alpha=0.5)
        ax.text(y = y+0.5, x=xmax, s=f"[{image_name}]Jpeg({a_quality:.2f}%)",
            horizontalalignment="right",
            fontdict={'fontsize': 12, 'fontweight': 'bold'})
        pass

    for a_quality in qualities:
        a_row = tmp_jpeg_image_df[tmp_jpeg_image_df["quality"] == a_quality].head(1)
        if a_row.shape[0] == 0: continue
        pnsr = a_row["psnr"].values[0]
        bpp = a_row["bpp"].values[0]

        x, y = bpp, pnsr
        ax.scatter(x, y, marker='D', color="yellow", s=150, edgecolors='black')
        ax.scatter(x, y, marker='.', color="red", edgecolors='black')
        # msg: str = f"JPG:{a_quality}%->({x:.2f},{y:.2f})"
        msg: str = f"JPG:{a_quality}%\n({x:.2f},{y:.2f})"
        ax.text(x=x, y=y, s = msg,
            horizontalalignment=kwargs["horizontalalignment"],
            fontdict={'fontsize': 14, 'fontweight': 'bold'})
        pass
    pass


def add_fixed_qat_points(a_df, ax, on_a_line_msg: bool = False,  pos_alignment: list = [], on_a_line = [], **kwargs):
    """Add fixed data points to a given input image."""
    if a_df.shape[0] == 0 or not ax: return

    if "horizontalalignment" not in kwargs.keys():
        kwargs["horizontalalignment"] = "center"
    if len(pos_alignment) == 0:
        n = len(a_df.groupby(by=["quant_techs_2", "nbits"]))
        pos_alignment = [kwargs["horizontalalignment"]] * n
        pass
    if len(on_a_line) == 0:
        n = len(a_df.groupby(by=["quant_techs_2", "nbits"]))
        on_a_line = [on_a_line_msg] * n
        pass

    for ii, (gk, gdata) in enumerate(a_df.groupby(by=["quant_techs_2", "nbits"])):
        tech, nbits = gk
        if nbits == 5: continue
        gdata_sorted = gdata.sort_values(by=["psnr"], ascending=False)
        a_row = gdata_sorted.head(1)
        psnr_val = a_row["psnr"].values[0]
        bpp = a_row["bpp"].values[0]

        x, y = bpp, psnr_val
        ax.scatter(x, y, marker='D', color="yellow", edgecolors='black')
        ax.scatter(x, y, marker='.', color="red", edgecolors='black')
        if on_a_line[ii]:
            msg: str = f"LRQ:{nbits:.0f}bits - ({bpp:.2f},{psnr_val:.2f})"
        else:
            msg: str = f"LRQ:{nbits:.0f}bits\n({bpp:.2f},{psnr_val:.2f})"
            pass
        ax.text(x=x, y=y+0.5, s = msg,
            # horizontalalignment=kwargs["horizontalalignment"],
            horizontalalignment=pos_alignment[ii],
            fontdict={'fontsize': 14, 'fontweight': 'bold'})
        pass
    pass


def add_fixed_prune_points(a_df, ax, **kwargs):
    """Add fixed data points to a given input image."""

    if a_df.shape[0] == 0 or not ax: return

    if "horizontalalignment" not in kwargs.keys():
        kwargs["horizontalalignment"] = "center"
        pass

    for gk, gdata in a_df.groupby(by=["image_name_2"]):
        gdata_sorted = gdata.sort_values(by=["psnr"], ascending=False)
        
        a_row = gdata_sorted.head(1)
        psnr_val = a_row["psnr"].values[0]
        bpp = a_row["bpp"].values[0]
        prune_rate = a_row["prune_rate"].values[0]

        x, y = bpp, psnr_val
        ax.scatter(x, y, marker='D', color="yellow", edgecolors='black', s=100)
        ax.scatter(x, y, marker='.', color="red", edgecolors='black')
        ax.text(x=x, y=y, s = f"{gk[0:]}:{prune_rate*100:.2f}%\n({bpp:.2f},{psnr_val:.2f})",
            horizontalalignment=kwargs["horizontalalignment"],
            fontdict={'fontsize': 12, 'fontweight': 'bold'})

        a_row = gdata_sorted.tail(1)
        psnr_val = a_row["psnr"].values[0]
        bpp = a_row["bpp"].values[0]
        prune_rate = a_row["prune_rate"].values[0]

        x, y = bpp, psnr_val
        ax.scatter(x, y, marker='D', color="yellow", edgecolors='black', s=100)
        ax.scatter(x, y, marker='.', color="red", edgecolors='black')
        ax.text(x=x, y=y, s = f"{gk[0:]}:{prune_rate*100:.2f}%\n({bpp:.2f},{psnr_val:.2f})",
            horizontalalignment=kwargs["horizontalalignment"],
            fontdict={'fontsize': 12, 'fontweight': 'bold'})

        pass
    pass


def show_models_choices(a_df, ax, **kwargs):
    """Show models choices."""

    if a_df.shape[0] == 0 or not ax: return

    if "horizontalalignment" not in kwargs.keys():
        kwargs["horizontalalignment"] = "center"
        pass

    x, y = a_df["bpp"].values, a_df["psnr"].values
    # plt.scatter(x, y, marker="^", s=200, label=f"Choices", color="red", edgecolors="black")
    plt.scatter(x, y, marker="*", s=350, label=f"Choices", color="red", edgecolors="black")

    cols = list(a_df.columns)
    for vals in a_df.values:
        
        vals_dict = dict(zip(cols, vals))
        
        bpp, psnr_val = vals_dict["bpp"], vals_dict["psnr"]
        x, y = vals_dict["bpp"], vals_dict["psnr"]
        prune_rate = vals_dict["prune_rate"]
        if str(prune_rate) != "-":
            prune_techs = vals_dict["prune_techs"]
            prune_techs = list(filter(lambda item: item.upper() == item or item == "+", prune_techs))
            prune_techs = ''.join(prune_techs)
            msg = f"Choice:\n{prune_techs}-{prune_rate*100:.2f}%\n({bpp:.2f},{psnr_val:.2f})"
            pass
        else:
            msg = f"Choice:\n({bpp:.2f},{psnr_val:.2f})"

        ax.text(x=x, y=y+1.0, s = msg,
            horizontalalignment=kwargs["horizontalalignment"],
            fontdict={'fontsize': 12, 'fontweight': 'bold'})
        pass
    pass
