from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

def get_table_agp_data(agp_df):
    headers = "compression,weights,bpp,psnr,file_size_bits,CR".split(",")
    table_1 = tabulate.tabulate(agp_df[headers], headers=headers, tablefmt="github")
    df_1 = agp_df[headers]
    return df_1, table_1

def get_table_jpge(result_jpeg_df):
    headers = "compression,quality,bpp,psnr,file_size_bits,CR".split(",")
    tmp_jpeg_df = result_jpeg_df[headers].tail(7)
    table_jpeg = tabulate.tabulate(tmp_jpeg_df.values, headers=tmp_jpeg_df.columns, tablefmt="github")
    return table_jpeg

def get_table_cameramen():
    data = dict(
        bpp=8.0040283203125,
        size=int(256*256*8),
        image_name='cameramen'
    )
    image_df = pd.DataFrame(data = [data], columns = 'bpp,size,image_name'.split(","), index=['cameramen'])
    tmp_img_df = image_df # .T
    # tmp_img_df.columns = "cameramen".split(",")
    # table_cameramen = tabulate.tabulate([list(tmp_img_df.values)], headers=tmp_img_df.columns, tablefmt="github")
    table_cameramen = tabulate.tabulate(list(tmp_img_df.values), headers=tmp_img_df.columns, tablefmt="github")
    return table_cameramen

def get_avg_stats_table(agp_df):
    headers = "compression,weights,bpp,psnr,file_size_bits,CR".split(",")
    mean_description = agp_df[headers].iloc[1:,:].mean(axis=0)
    table_2 = tabulate.tabulate([list(mean_description.values)], headers=mean_description.index, tablefmt="github")
    return mean_description, table_2

def show_summary_data_agp(agp_df, result_jpeg_df):
    
    # table_cameramen = get_table_cameramen()
    # table_jpeg = get_table_jpge(result_jpeg_df)
    
    mean_description, table_2 = get_avg_stats_table(agp_df)
    df_1, table_1 = get_table_agp_data(agp_df)
    
    # create output widgets
    widget1 = widgets.Output()
    widget2 = widgets.Output()
    widget3 = widgets.Output()
    widget4 = widgets.Output()

    # render in output widgets
    with widget1:
        df_1["weights"] = np.ceil(df_1["weights"].values).astype(dtype=np.int)
        df_1["file_size_bits"] = np.ceil(df_1["file_size_bits"].values).astype(dtype=np.int)
        display(df_1.sort_values(["bpp"], ascending=False).head(7).style.hide_index())
    with widget2:
        mean_description_df = pd.DataFrame(data = list(zip(mean_description.index, list(mean_description))), columns='Avg-Stats-Siren,Value'.split(","))
        display(mean_description_df.style.hide_index())
    with widget3:
        data = dict(
            bpp=8.0040283203125,
            size=int(256*256*8),
        )
        image_df = pd.DataFrame(data = [data], )
        tmp_img_df = image_df.T
        tmp_img_df.columns = "cameramen".split(",")
        # display(tmp_img_df.style.hide_index())
        display(tmp_img_df.style)
    with widget4:
        headers = "compression,quality,bpp,psnr,file_size_bits,CR".split(",")
        display(result_jpeg_df[headers].tail(7).sort_values(["quality"], ascending=False).style.hide_index())

    sidebyside = widgets.HBox([widget1, widget2, widget3, widget4])
    display(sidebyside)
    return
