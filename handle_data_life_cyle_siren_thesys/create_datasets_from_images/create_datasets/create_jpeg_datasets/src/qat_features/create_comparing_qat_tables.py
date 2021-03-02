from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

# import dataframe_image as dfi

def create_delta_attributes(table_df:pd.DataFrame, a_delta_column:str):
    """TODO comment it."""
    table_df[f'delta_{a_delta_column}'] = table_df[f'{a_delta_column}'].values - table_df[f'{a_delta_column}'].values[0]
    table_df[f'delta_{a_delta_column}(%)'] = (table_df[f'delta_{a_delta_column}'].values * (-1) / table_df[{a_delta_column}].values[0]) * 100
    table_df[f'delta_{a_delta_column}(%)'][0] = 0.0
    return 


def create_comparing_results_table(
    table_df:pd.DataFrame, show_columns:list,
    delta_columns:list, formatter_dict:dict,
    extend_formatter_keys:list, hide_date_init_from:bool=False,
    show_deltas:bool= True, hide_index:bool=True,
    enable_cmap:bool=False, cmap:str='viridis'):
    """TODO comment it."""
    if hide_date_init_from:
        drop_date_init_from = "date,init-from".split(",")
        show_columns = list(filter(lambda item: item not in drop_date_init_from, show_columns))
    if show_deltas:
        for a_delta_column in delta_columns:
            create_delta_attributes(table_df, a_delta_column)
            pass
    else:
        deltas_list = list(filter(lambda item: item.startswith('delta_'), list(table_df.columns)))
        if len(deltas_list) != 0:
            show_columns = list(filter(lambda item: item not in deltas_list, show_columns))
        # table_df = table_df.drop(deltas_list, axis = 1)
    for a_key in extend_formatter_keys:
        formatter_dict[a_key] = '{:.2f}'
        pass
    
    if hide_index:
        if enable_cmap:
            resulting_table_df = table_df[show_columns]\
                .style\
                .format(formatter_dict)\
                .background_gradient(cmap=f'{cmap}').hide_index()
        else:
            resulting_table_df = table_df[show_columns]\
                .style\
                .format(formatter_dict)\
                .hide_index()
    else:
        if enable_cmap:
            resulting_table_df = table_df[show_columns]\
                .style\
                .format(formatter_dict)\
                .background_gradient(cmap=f'{cmap}')
        else:
            resulting_table_df = table_df[show_columns]\
                .style\
                .format(formatter_dict)
    
    return resulting_table_df


def show_scatter_plot_2(qat_df:pd.DataFrame, jpeg_df:pd.DataFrame):
    """TODO comment it."""
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
                 title=f'{y.capitalize()} vs {x.capitalize()}',
                theme=theme, colorscale=colorscale, categories="cmprss-class-3")
            else:
                qat_df[(qat_df["bpp"] <= 12.0)].iplot(kind='scatter', x=x, y=y, mode='markers', 
                 xTitle=x.capitalize(), yTitle=y.capitalize(),
                 text='cmprss-class',
                 title=f'{y.capitalize()} vs {x.capitalize()}',
                theme=theme, colorscale=colorscale, categories="cmprss-class-3")
            pass
        pass
    pass
