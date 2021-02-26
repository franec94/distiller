#!/usr/bin/env python3
# -*- enc:utf-8 -*-

from src.libs.project_libs import *
from src.utils.create_jpeg_dataset_parser import get_create_jpeg_dataset_parser

# ============================================ #
# Setup Script Task
# ============================================ #
def setup_script(args) -> dict:
    """Setup script.\n
    Args:
    -----
    `args` - Namespace object.\n
    Returns:
    --------
    `conf_data` -dict.\n
    """
    conf_data = dict()
    if args.conf_file:
        conf_data = read_conf_file(conf_file_path=args.conf_file)
        pass
    
    if "input_image" not in conf_data.keys():
        conf_data["input_image"] = args.input_image
        pass
    if "output_dir" not in conf_data.keys():
        conf_data["output_dir"] = args.output_dir
        pass
    
    
    conf_data["image_name"] = "cameramen"
    image_name = conf_data["input_image"]
    if image_name:
        image_name = os.path.basename(image_name)
        image_name, _ = os.path.splitext(image_name)
        conf_data["image_name"] = image_name
        pass
    
    conf_data["timestamp"] = time.time()
    path_pieces: list = [conf_data["output_dir"], conf_data["image_name"], f"out_" + str(conf_data["timestamp"])]
    conf_data["out_dir_ts"] = os.path.join(*path_pieces)

    create_dir(dir_path=conf_data["out_dir_ts"])
    if "logs_dir" not in conf_data.keys():
        conf_data["logs_dir"] = os.path.join(conf_data["out_dir_ts"], f"logs")
    create_dir(dir_path=conf_data["logs_dir"])
    
    get_root_level_logger(root_path=conf_data["logs_dir"], loggger_name='create_jpeg_dataset.log')
    logging.info("Script's Setup phase is proceeding...")
    logging.info("Created root dir {}".format(conf_data["out_dir_ts"]))
    logging.info("Created logging dir {}".format(conf_data["logs_dir"]))

    
    if "images_dir" not in conf_data.keys():
        conf_data["images_dir"] = os.path.join(conf_data["out_dir_ts"], f"images")
    logging.info("Creating {}...".format(conf_data["images_dir"]))
    create_dir(dir_path=conf_data["images_dir"])

    if "configs_dir" not in conf_data.keys():
        conf_data["configs_dir"] = os.path.join(conf_data["out_dir_ts"], f"configs")
    logging.info("Creating {}...".format(conf_data["configs_dir"]))
    create_dir(dir_path=conf_data["configs_dir"])


    if "csv_dir" not in conf_data.keys():
        conf_data["csv_dir"] = os.path.join(conf_data["out_dir_ts"], f"out_csv")
    logging.info("Creating {}...".format(conf_data["csv_dir"]))
    create_dir(dir_path=conf_data["csv_dir"])

    conf_data["images_list"] = list()
    conf_data["script_stats"] = dict()
    logging.info("Setup script ended.")

    return conf_data


# ============================================ #
# Create Images 
# ============================================ #
def create_table_as_image(image_name:str, im:PIL.Image, im_cropped:PIL.Image, conf_data: dict) -> str:
    if "script_stats" in conf_data.keys():
        script_stats = conf_data["script_stats"]
        no_images = script_stats.setdefault("no_images", 0)
        script_stats["no_images"] = no_images + 2
        pass
    image = im
    data_table: dict = dict(
            name=f"{image_name}",
            shape=image.size,
            size_byte=image.size[0]*image.size[1],
            image_band=image.getbands(),
            entropy=image.entropy(),
    )
    image = im_cropped
    data_table_2: dict = dict(
        name=f"{image_name}",
        shape=image.size,
        size_byte=image.size[0]*image.size[1],
        image_band=image.getbands(),
        entropy=image.entropy(),
    )
    a_table = pd.DataFrame(data=[data_table, data_table_2], index = "full,cropped".split(","))
    images_dir = conf_data["images_dir"]
    fig_name_path_tb = os.path.join(images_dir, "a_df_table.png")

    # fig_name_path_tb = "/home/franec94/Desktop/a_df_table.png"
    dfi.export(a_table, fig_name_path_tb)
    return fig_name_path_tb


def create_summary_grid_plot_image(conf_data: dict, image_file_path: str = None, crop_size = 256):
    if not image_file_path:
        image_name = "Cameramen"
        pass
    else:
        image_name, _ = os.path.splitext(
            os.path.basename(image_file_path)
        )
        pass
    
    im = load_target_image(image_file_path = image_file_path)
    if not isinstance(crop_size, list):
        crop_size = (crop_size,crop_size)
    im_cropped = get_cropped_by_center_image(im, target = crop_size)
    
    fig_name_path_tb = create_table_as_image(image_name, im, im_cropped, conf_data)
    conf_data["images_list"].append(fig_name_path_tb)
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2,2, 1)
    imgplot = plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Image: {image_name.capitalize()} - {im.size} pixel size", fontweight="bold", fontsize=15)

    ax = fig.add_subplot(2,2, 2)
    imgplot = plt.imshow(im_cropped, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Image: {image_name.capitalize()} - {im_cropped.size} pixel size",
                 fontweight="bold", fontsize=15)


    ax = fig.add_subplot(2,2, 3)

    ax.plot(im.histogram(), label = f"{image_name.capitalize()} - Full ({im.size[0]}x{im.size[1]})")
    ax.plot(im_cropped.histogram(), label = f"{image_name.capitalize()} - Cropped ({im_cropped.size[0]}x{im_cropped.size[1]})")

    ax.set_xlabel("Pixel Intensity (Grayscale Image 0-255 levels)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Density Histogram:\n{image_name.capitalize()} ({im.size[0]}x{im.size[1]}) vs. Cropped ({im_cropped.size[0]}x{im_cropped.size[1]})",
        fontweight="bold", fontsize=15)

    plt.legend()
    plt.grid(True)

    ax = fig.add_subplot(2,2, 4)
    img_tb = mpimg.imread(f"{fig_name_path_tb}")
    imgplot = plt.imshow(img_tb)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(
        f"Summary Table Image {image_name.capitalize()}:\nfull size and cropped versions",
        fontweight="bold", fontsize=15)

    images_dir = conf_data["images_dir"]
    fig_name_path = os.path.join(images_dir, "complex.png")
    plt.savefig(fig_name_path)
    conf_data["images_list"].append(fig_name_path)
    # plt.show();
    plt.close()
    pass


def create_scatterplot_jpeg_dataset(conf_data:dict, a_df:pd.DataFrame, x:str="bpp", y:str="psnr", **kwargs) -> str:
    """Create scatterplot image for jpeg dataset."""


    if "script_stats" in conf_data.keys():
        script_stats = conf_data["script_stats"]
        no_images = script_stats.setdefault("no_images", 0)
        script_stats["no_images"] = no_images + 1
        pass

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (10, 10)
    if "xlable" not in kwargs.keys():
        if x.lower() == "psnr":
            xlabel = "Psnr [db]"
            kwargs["xlable"] = xlabel
        else: kwargs["xlable"] = x.capitalize()
        pass
    if "ylable" not in kwargs.keys():
        if x.lower() == "psnr":
            ylabel = "Psnr [db]"
            kwargs["ylable"] = xlabel
        else: kwargs["ylable"] = x.capitalize()
        pass
    if "title" not in kwargs.keys():
        xlabel, ylabel = kwargs["xlable"], kwargs["ylable"]
        title = f"{xlabel} vs. {ylabel} (Jpeg Compression)"
        kwargs["title"]= title
        pass
    fig, ax = plt.subplots(1, 1, figsize=kwargs["figsize"])
    sns.scatterplot(data=a_df, x=f"{x}", y=f"{y}", ax=ax)

    xlabel, ylabel = kwargs["xlable"], kwargs["ylable"]
    images_dir = conf_data["images_dir"]
    scatterplot_filename = f"{xlabel}_vs_{ylabel}.png"
    scatterplot_filepath = os.path.join(images_dir, scatterplot_filename)

    plt.title(kwargs["title"])
    plt.ylabel(kwargs["xlable"])
    plt.ylabel(kwargs["xlable"])
    plt.legend()
    plt.savefig(scatterplot_filepath)
    plt.close()
    return scatterplot_filepath


def create_image(conf_data, target_image, image_file_path) -> str:
    """Create image plot and save it."""
    if image_file_path:
        filename = os.path.basename(image_file_path)
    else:
        filename = "Cameramen"
        pass

    h, w = target_image.size
    image_path = os.path.join(
        conf_data["images_dir"], f"{filename.lower()}_{h}h_{w}w.png")

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(target_image,
            cmap='gray', vmin = 0, vmax = 255,interpolation='none')
    plt.title(f"Image: {filename}")
    plt.xlabel(f"Image size: {target_image.size} (WxH)")
    plt.savefig(image_path)
    plt.close()
    return image_path


def create_image_details(conf_data: dict, image_file_path, create_out_images: bool = False) -> PIL.Image:
    """Create original image details."""
    target_image: PIL.Image = load_target_image(image_file_path=image_file_path)

    
    if create_out_images:
        image_path = create_image(conf_data=conf_data, target_image=target_image, image_file_path=image_file_path)
        conf_data["images_list"].append(image_path)
        if image_file_path:
            filename = os.path.basename(image_file_path)
        else:
            filename = "Cameramen"
            pass

        h, w = target_image.size
        image_path = os.path.join(
            conf_data["images_dir"], f"hist_{filename.lower()}_{h}h_{w}w.png")
        fig = get_histogram_image(image=target_image)
        plt.savefig(image_path)
        conf_data["images_list"].append(image_path)
        plt.close()
        if "script_stats" in conf_data.keys():
            script_stats = conf_data["script_stats"]
            no_images = script_stats.setdefault("no_images", 0)
            script_stats["no_images"] = no_images + 2
            pass
        pass

    return target_image


def create_cropped_image_details(conf_data: dict, image_file_path, target_crop = 256, create_out_images:bool = False) -> None:
    """Create cropped by its center input image."""
    target_image: PIL.Image = load_target_image(image_file_path=image_file_path)

    cropped_image = \
        get_cropped_by_center_image(im=target_image, target=target_crop)

    if create_out_images:
        image_path = create_image(conf_data=conf_data, target_image=cropped_image, image_file_path=image_file_path)
        conf_data["images_list"].append(image_path)

        if image_file_path:
            filename = os.path.basename(image_file_path)
        else:
            filename = "Cameramen"
            pass

        h, w = cropped_image.size
        image_path = os.path.join(
            conf_data["images_dir"], f"hist_{filename.lower()}_{h}h_{w}w.png")
        fig = get_histogram_image(image=cropped_image)
        conf_data["images_list"].append(image_path)
        plt.savefig(image_path)
        plt.close()
        
        if "script_stats" in conf_data.keys():
            script_stats = conf_data["script_stats"]
            no_images = script_stats.setdefault("no_images", 0)
            script_stats["no_images"] = no_images + 2
            pass
        pass

    return cropped_image

# ============================================ #
# Load Images (Origina, Cropped)
# ============================================ #
def load_original_image(image_file_path) -> PIL.Image:
    target_image: PIL.Image = load_target_image(image_file_path=image_file_path)
    return target_image


def load_cropped_image(image_file_path, target_crop = 256) -> PIL.Image:
    target_image: PIL.Image = load_target_image(image_file_path=image_file_path)
    return target_image


# ============================================ #
# Script Main Task (Create Dataset)
# ============================================ #
def create_out_dataset(conf_data: dict, target_image) -> pd.DataFrame:
    """Create output dataset."""
    _, compressed_file_size_bits = calculate_cropped_image_size(image=target_image)

    qualities = np.arange(20, 95+1)
    result_tuples, failure_qualities = calculate_several_jpeg_compression(
        image=target_image,
        image_dim_bits=compressed_file_size_bits,
        qualities=qualities
    )
    if len(failure_qualities) != 0:
        print(f"Some qualities have failed")
        pass
    jpeg_df = pd.DataFrame(data=result_tuples)

    csv_filename = os.path.join(conf_data["csv_dir"], "dataset.csv")
    jpeg_df.to_csv(csv_filename)
    return jpeg_df


# ============================================ #
# Script Last Tasks (Summary Reports)
# ============================================ #
def summary_script_runned(conf_data):
    data_tb = dict(
        out_dir=conf_data["out_dir_ts"],
        elapsed_time=conf_data["elapsed_time"],
    )
    if "script_stats" in conf_data.keys():
        script_stats = conf_data["script_stats"]
        no_images = script_stats.setdefault("no_images", 0)
        data_tb["no_images"] = no_images
        pass
        
    meta_data_tb = dict(
        tabular_data=data_tb.items()
    )
    table = tabulate.tabulate(**meta_data_tb)
    print(table)
    return table


def save_all_images_as_merged_pdf(conf_data: dict) -> None:
    """Comment it."""

    output_dir_path = conf_data["out_dir_ts"]
    figures_list: list = conf_data["images_list"]
    if len(figures_list) == 0: return

    pdf_filename =  os.path.join(
        output_dir_path, "merged.pdf")
    
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

# ============================================ #
# Entry Point (MAIN)
# ============================================ #
def main(args) -> None:
    """Main function."""

    conf_data: dict = setup_script(args=args)
    logging.info("Start script, after initial setup phase.")
    start_time_script = time.time()

    logging.info("Loading original image...")
    _ = create_image_details(conf_data=conf_data,
        image_file_path=conf_data["input_image"], create_out_images=True)
    logging.info("Loading cropped image...")
    cropped_image = create_cropped_image_details(conf_data=conf_data,
        image_file_path=conf_data["input_image"], target_crop=256, create_out_images=True)

    
    logging.info("Creating output dataset...")
    jpeg_df = create_out_dataset(conf_data=conf_data, target_image=cropped_image)

    x="bpp"; y="psnr"
    logging.info(f"Creating scatterplot {x} vs {y}...")
    image_path = create_scatterplot_jpeg_dataset(conf_data=conf_data,
        a_df = jpeg_df, x=f"{x}", y=f"{y}"
    )
    conf_data["images_list"].append(image_path)

    create_summary_grid_plot_image(
        conf_data=conf_data,
        image_file_path=conf_data["input_image"],
        crop_size = 256)

    end_time_script = time.time()
    elapsed_time = end_time_script - start_time_script
    logging.info(f"Elapsed time: {elapsed_time:.2f} sec.")

    if len(conf_data["images_list"]) != 0:
        logging.info("Merging all created images...")
        save_all_images_as_merged_pdf(conf_data=conf_data)
        pass

    conf_data["elapsed_time"] = elapsed_time
    logging.info("Summary Scritp Statistics:")
    table = summary_script_runned(conf_data=conf_data)
    logging.info(f"\n{str(table)}")

    
    
    
    pass


if __name__ == "__main__":
    parser: argparse.ArgumentParser = \
        get_create_jpeg_dataset_parser()
    args = parser.parse_args()
    main(args=args)
    pass
