from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from tqdm.autonotebook import tqdm

# =========================================================== #
# Utils Functions
# =========================================================== #
def get_histogram_image(image: PIL.Image, **kwargs):
    """Get histogram image.
    Args:
    -----
    `iamge` - PIL.Image.\n
    Returns:
    --------
    `fig` - matplotlib fig object.\n
    """
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (10, 10)
    if "title" not in kwargs.keys():
        kwargs["title"] = f"Histogram Camera Image({image.size[0]}x{image.size[1]})"
        pass
    fig = plt.figure(figsize=kwargs["figsize"])
    _ = plt.plot(image.histogram())
    plt.title(kwargs["title"])
    return fig


def get_new_targets(target, size):
    """Get coordinates of new cropped target image.
    Args:
    -----
    `target` - int, cropping size with respect center of image.\n
    `size` - int, original image size.\n
    Returns:
    --------
    `extreme_1` - int, position of cropped image.\n
    `extreme_2` - int, position of cropped image.\n
    """
    offset = target // 2
    if target % 2 == 0:
        extreme_1 = size // 2
        residual = 0
    else:
        extreme_1 = size // 2 - 1
        residual = 1
        pass
    extreme_2 = size // 2
    return extreme_1 - offset + residual, extreme_2 + offset + residual


def get_image_details_as_table(image: PIL.Image):
    """Get image data inforamtion as tabulate.tabulate output table.\n
    Args:
    -----
    `image` - PIL.Image.\n
    Returns:
    --------
    `table` - output table.\n
    """
    data_table = dict(
    name="Camera",
    shape=image.size,
    size_byte=image.size[0]*image.size[1],
    image_band=image.getbands(),
    )
    meta_data_table = dict(
        tabular_data=data_table.items(),
        tablefmt="pipe" # "github"
    )
    table = tabulate.tabulate(**meta_data_table)
    # print(table)
    return table


def get_cropped_by_center_image(im, target = 25):
    """Get Cropped Image by its center
    Args:
    -----
    `im` - - PIL.Image.\n
    `target` - int or list, cropping size with respect center of image.\n
    Returns:
    --------
    `im_cropped` - PIL.Image.\n
    """
    width, height = im.size

    if isinstance(target, int):
        target = (target, target)
        pass

    left, right = get_new_targets(target[0], width)
    top, bottom = get_new_targets(target[1], height)

    # print(im.crop((left, top, right, bottom)).size)
    # print((left, top, right, bottom))

    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped


# =========================================================== #
# Business Functions (Jpeg Compression Tasks)
# =========================================================== #

def get_jpeg_namedtuple_builder() -> (object, list):
    """Get Jpeg Namedtuple Builder.\n
    Returns:
    --------
    `WeightsPsnr` - namedtuple name.\n
    """
    typename: str = 'JpegData'
    fields_name: list = [
        'mse', 'psnr', 'ssim',
        'quality', 'size_byte', 'bpp',
        'width', 'height', 'CR', 'image_name',
        'cmprss_class', 'cmprss_class_2', 'cmprss_class_3',
        'prune_techs', 'prune_rate', 'quant_tech', 'nbits'
        ]
    fields_name_2: list = [
        'mse', 'psnr', 'ssim',
        'quality', 'size_byte', 'bpp',
        'width', 'height', 'CR', 'image_name',
        'cmprss-class', 'cmprss-class-2', 'cmprss-class-3',
        'prune_techs', 'prune_rate', 'quant_tech', 'nbits'
        ]
    JpegData = collections.namedtuple(typename, fields_name) 
    return JpegData, fields_name_2


def calculate_cropped_image_size(image, quality = None, ext = None) -> (PIL.Image, float):
    """Calculater cropped image size.
    Returns:
    --------
    `im_jpeg` - PIL.Image, compressed image instance.\n
    `compressed_file_size_bits` - float, compressed size.\n
    """
    im_jpeg = None
    with BytesIO() as f:
        # im_tmp.save(f'myimg.jpg', quality = int(quality))
        # im_jpeg = Image.open('myimg.jpg')

        if ext:
            image.save(f, format=f'{ext.upper()}', quality = int(quality))
        else:
            image.save(f, format='PNG')
        f.seek(0)
        compressed_file_size_bits = f.getbuffer().nbytes * 8
        im_jpeg = copy.deepcopy(PIL.Image.open(f))
        assert im_jpeg.size == image.size, "im_jpeg.size != image.size"
    return im_jpeg, compressed_file_size_bits


def calculate_cmprss_class_labels(data_dict: collections.OrderedDict) -> collections.OrderedDict:
    quality = data_dict["quality"]
    data_dict["cmprss-class"] = f"Jpeg:{quality:.2f}"
    data_dict["cmprss-class-2"] = "Jpeg"
    data_dict["cmprss-class-3"] = "Jpeg"
    return data_dict


def calculate_prune_quant_labels(data_dict: collections.OrderedDict) -> collections.OrderedDict:
    data_dict["prune_techs"] = f"Jpeg"
    data_dict["quant_tech"] = f"Jpeg"
    return data_dict


def calculate_scores(image, im_jpeg, image_dim_bits, compressed_file_size_bits, quality, data_dict: collections.OrderedDict) -> list:
    """Calculate scores.
    Returns:
    --------
    `values` - list object.\n
    """

    error_message = f"len(np.asarray(image)) != len(np.asarray(im_jpeg)), acctually, {len(np.asarray(image))} != {len(np.asarray(im_jpeg))}"
    assert len(np.asarray(image)) == len(np.asarray(im_jpeg)), error_message

    # Data used for deriving scores
    width, height = im_jpeg.size[0], im_jpeg.size[1]
    pixels = width * height

    # Scores
    bpp = compressed_file_size_bits / pixels
    mse_score = mean_squared_error(np.asarray(image), np.asarray(im_jpeg))
    psnr_score = psnr(np.asarray(image), np.asarray(im_jpeg), data_range=255)
    ssim_score = ssim(np.asarray(image), np.asarray(im_jpeg), data_range=255)
    CR = image_dim_bits / compressed_file_size_bits

    # Store results into a list
    values: list = [mse_score, psnr_score, ssim_score, quality, compressed_file_size_bits / 8, bpp, width, height, CR, quality, 32]
    keys: list = ["mse", "psnr", "ssim", "quality", "size_byte", "bpp", "width", "height", "CR", "quality", "nbits"]
    for k, v in zip(keys, values):
        data_dict[k]=v
        pass
    return values


def calculate_several_jpeg_compression(image: PIL.Image, image_dim_bits:int, qualities:list, image_name: str) -> (list, list):
    """Calculate several jpeg compression instances.
    Returns:
    --------
    `result_tuples` - list of tuples with stored data related to tested jpeg compression levels.\n
    `failure_qualities` - list of failures for qualities that have not been calculated.\n
    """
    
    # Named tuple for creating a record related to a trial for compressing the target image.
    JpegData, fields_name = get_jpeg_namedtuple_builder()

    # List used to save results and keep trace of failures, if any.
    result_tuples: list = []
    failure_qualities: list = []

    # Then test the effect of several different quality values used in compression transform.
    with tqdm(total=len(qualities)) as pbar:
        for quality in qualities:
            try:
                # Convert to JPEG specifying quality of compression.
                data_dict = collections.OrderedDict(zip(fields_name, [None] * len(fields_name)))
                im_jpeg, compressed_file_size_bits = \
                    calculate_cropped_image_size( \
                        image=image, quality=quality, ext = 'JPEG')
                # pbar.write(f"Targte Image Size: {image.size}")
                # pbar.write(f"Compressed Image Size: {im_jpeg.size}")
        
                # Calculate quantities to be stored for this trial
                values = \
                    calculate_scores( \
                        image=image, im_jpeg=im_jpeg,
                        image_dim_bits=image_dim_bits,
                        compressed_file_size_bits=compressed_file_size_bits,
                        quality=quality,
                        data_dict=data_dict)
                # values = values + [image_name, quality]
                data_dict["image_name"] = image_name

                calculate_cmprss_class_labels(data_dict=data_dict)
                calculate_prune_quant_labels(data_dict=data_dict)

                # Append result
                # result_tuples.append(JpegData._make(values))
                # values = list(data_dict.values())
                # result_tuples.append(JpegData._make(values))
                result_tuples.append(data_dict)
            except Exception as err:
                # Keep track of unaccepted quality values for compressing the image
                pbar.write(f"{str(err)}")
                failure_qualities.append(quality)
                pass
            pbar.update(1)
            pass
        pass
    return result_tuples, failure_qualities


def calculate_several_jpeg_compression_with_crops(image: PIL.Image, qualities: list, cropping_list: list) -> (list, list):
    """Calculate several jpeg compression instances."""
    # Named tuple for creating a record related to
    # a trial for compressing the target image.
    JpegData = get_jpeg_namedtuple_builder()

    # List used to save results and keep trace of failures, if any.
    result_tuples = []
    failure_qualities = []

    # Gather results.
    for edges in cropping_list: # for edges in edges_list:
    
        # Firstly crop image to desired shape.    
        left, top, right, bottom = list(map(int, edges))
        im_tmp = image.crop((left, top, right, bottom))
    
        # Get size cropped image
        cropped_file_size_bits = None
        with BytesIO() as f:
            im_tmp.save(f, format='PNG')
            cropped_file_size_bits = f.getbuffer().nbytes * 8
            pass
    
        # Then test the effect of several different quality values
        # used in compression transform.
        for quality in qualities:
            try:
                # Convert to JPEG specifying quality of compression.
                with BytesIO() as f:
                    # im_tmp.save(f'myimg.jpg', quality = int(quality))
                    # im_jpeg = Image.open('myimg.jpg')
                
                    im_tmp.save(f, format='JPEG', quality = int(quality))
                    f.seek(0)
                    im_jpeg = PIL.Image.open(f)
                    assert im_jpeg.size == im_tmp.size, "im_jpeg.size != im_tmp.size"
    
                    # Calculate quantities to be stored for this trial
            
                    # data used for deriving scores
                    width, height = im_jpeg.size[0], im_jpeg.size[1]
                    pixels = width * height
                    # compressed_file_size_bits = Path('myimg.jpg').stat().st_size * 8
                    compressed_file_size_bits = f.getbuffer().nbytes * 8
            
                    # Scores
                    bpp = compressed_file_size_bits / pixels    
                    psnr_score = psnr(np.asarray(im_tmp), np.asarray(im_jpeg), data_range=255)
                    ssim_score = ssim(np.asarray(im_tmp), np.asarray(im_jpeg), data_range=255)
                    CR = cropped_file_size_bits / compressed_file_size_bits
            
                    # Store results into a list
                    values = [psnr_score, ssim_score, quality, compressed_file_size_bits, bpp, width, height, CR]
                    result_tuples.append(JpegData._make(values))
                    pass
            except Exception as err:
                # Keep track of unaccepted quality values for compressing the image
                print(err)
                failure_qualities.append(quality)
            pass
        pass
    return result_tuples, failure_qualities

# =========================================================== #
# Business Functions (Jpeg Dataset Creation Tasks)
# =========================================================== #
def get_ready_siren_df_to_merge(siren_df, image):
    with BytesIO() as f:
        image.save(f, format='PNG')
        file_size_bits = f.getbuffer().nbytes * 8
        pass
    # Define "BPP attribute" and add it to existing df.
    siren_df['bpp'] = siren_df['#params'].values * 32 / (image.size[0] * image.size[1])
    
    # Define "file_size_bits attribute" and add it to existing df.
    siren_df['file_size_bits'] = siren_df['#params'].values * 32
    # Define "CR attribute" and add it to existing df.
    siren_df['CR'] = file_size_bits / (siren_df['#params'].values * 32)
    
    # Define "Compression Label" and add it to existing df.
    hf_arr = siren_df['hf'].values.astype(dtype = np.int)
    create_label_lambda = lambda hf: f'siren-{hf}'
    siren_df['compression'] = list(map(create_label_lambda, hf_arr))
    
    return siren_df


def get_ready_jpeg_df_to_merge(jpeg_df):
    # Define "Compression Label" and add it to existing df.
    jpeg_df['compression'] = ['jpeg'] * jpeg_df.shape[0]
    return jpeg_df


def prepare_and_merge_target_dfs(siren_df, jpeg_df, *args, **kwargs):
    # pprint(kwargs)
    
    # Target image, either cropped or not, depending on the kind of run.
    image = kwargs['image']
    
    # Prepare siren_df for merging.
    siren_df = get_ready_siren_df_to_merge(siren_df, image=image)
    
    # Prepare jpeg_df for merging.
    jpeg_df = get_ready_jpeg_df_to_merge(jpeg_df)
    
    # Get columns to be merged and new columsn for
    # rename them after merging.
    siren_columns_for_merge = kwargs['siren_columns_for_merge']
    jpeg_columns_for_merge = kwargs['jpeg_columns_for_merge']
    columns_names_merge = kwargs['columns_names_merge']
    
    # Performe merging.
    data_frames_list = [
        siren_df[siren_columns_for_merge],
        jpeg_df[jpeg_columns_for_merge],
    ]
    merged_df = pd.concat(data_frames_list, names = columns_names_merge)
    return merged_df, siren_df, jpeg_df
