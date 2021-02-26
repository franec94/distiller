from src.libs.std_python_libs import *
from src.libs.data_science_libs import *
from src.libs.graphics_and_interactive_libs import *

from src.libs.all_end_to_end_exp_analyses import *

from src.handle_raw_data.load_data_from_conf import get_reference_datasets
from src.graphics.create_graphics_for_report import * # show_pie_chart

from src.create_report.create_report import get_read_of_cols
from src.create_report.create_report import save_all_images_as_merged_pdf
from src.create_report.create_report import check_graphics_2pie_2bars
from src.create_report.create_report import check_graphics_complex_charts
from src.create_report.create_report import check_graphics_complex_charts_pbbv
from src.create_report.create_report import create_table_via_groupby
from src.create_report.create_report import creat_table_for_bar_plot_counts_ws_index
from src.create_report.create_report import fecth_all_report_datasets
from src.create_report.create_report import get_initial_table_jpeg

import PyPDF2
import img2pdf
import fitz

