#!/usr/bin/env python3 -f
#-*- enc:utf-8 -*-

from src.libraries.all_libs import *
from src.generic.custom_cmd_line_parsers.custom_parser_test_batch import get_custom_parser_test_batch


def merge_datasets(models_df, res_test_df, time_stamp = None, args = None):
    if not type(res_test_df) == pd.core.frame.DataFrame:
        print("No merging is done!")
        return

    dest_columns = "date_test,mse,psnr,ssim".split(",")
    src_columns = "date,MSE,PSNR,SSIM".split(",")
    models_copy_df = copy.deepcopy(models_df)
    for a_dest, a_src in list(zip(dest_columns, src_columns)):
        models_copy_df[f'{a_dest}'] = res_test_df[f'{a_src}'].values
        pass
    
    print(models_copy_df.head(5))
    print(models_copy_df.info())
    if time_stamp is None:
        time_stamp = time.time()
        pass
    models_copy_path = os.path.join(f"{args.output_dataset_path}", f"out_merged_{time_stamp}.csv")
    models_copy_df.to_csv(models_copy_path)
    pass


def run_and_wait_subprocess(cmd, show_subprocess_info=False):
    process = subprocess.Popen(cmd, shell=True,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode
    if show_subprocess_info:
        print(out.decode())
        print(err.decode())
        print(errcode)
        pass
    pass


def show_res_tests_df_info(results_csv_path):
    if os.path.exists(results_csv_path) and os.path.isfile(results_csv_path):
        res_test_df = pd.read_csv(results_csv_path)
        print(res_test_df.head(5))
        print(res_test_df.info())
        return res_test_df
    else:
        print(f"File {results_csv_path} not found!")
    return None


def run_tests_in_batch(args, models_df, verbose = 0):
    LOGGING_ROOT = f"{args.tests_logging_root}"
    time_stamp = time.time()
    RESULTS_CSV_PATH = os.path.join(f"{args.output_dataset_path}", f"out_{time_stamp}.csv")

    table_meta = dict(
        tabular_data = zip("LOGGING_ROOT,RESULTS_CSV_PATH,NO TESTS".split(","), [LOGGING_ROOT, RESULTS_CSV_PATH, models_df.shape[0]]),
    )
    table = tabulate.tabulate(**table_meta)
    print(table)
    columns = list(models_df.columns)
    for a_row in tqdm.tqdm(models_df.values, total = models_df.shape[0]):
        a_row_dict = dict(zip(columns, a_row))
        STATE_DICT_MODEL_FILE = os.path.join(a_row_dict['root_dir'], a_row_dict['model_name'])
        SIDELENGTH = a_row_dict['sidelength']
        N_HF = a_row_dict["n_hf"]
        N_HL = a_row_dict["n_hl"]

        if verbose == 1:
            tqdm.tqdm.write(f"Processing '{STATE_DICT_MODEL_FILE}'...")
        if not os.path.exists(STATE_DICT_MODEL_FILE):
            tqdm.tqdm.write(f"File '{STATE_DICT_MODEL_FILE}' not exists!")
            pass
        if not os.path.isfile(STATE_DICT_MODEL_FILE):
            tqdm.tqdm.write(f"Resources '{STATE_DICT_MODEL_FILE}' is not a file!")
            pass
        

        # cmd=f"""ls -la {a_row_dict['root_dir']}"""        
        cmd=f"""
CUDA_VISIBLE_DEVICES=0 \
python3 siren_main_app.py \
    --logging_root {LOGGING_ROOT} \
    --experiment_name 'test_model' \
    --sidelength {SIDELENGTH} \
    --n_hf {N_HF}  \
    --n_hl {N_HL} \
    --seed 0 \
    --cuda \
    --evaluate \
    --exp-load-weights-from {STATE_DICT_MODEL_FILE} \
    --save_test_data_to_csv_path {RESULTS_CSV_PATH} \
    --verbose 0
        """
        # print(cmd)
        run_and_wait_subprocess(cmd, show_subprocess_info=False)
        
        pass

    res_test_df = show_res_tests_df_info(results_csv_path=RESULTS_CSV_PATH)
    return time_stamp, RESULTS_CSV_PATH, res_test_df


def check_cmd_line_args(args):
    if not os.path.exists(args.input_dataset_file):
        print(f"Error: '{args.input_dataset_file}' not exists", file=sys.stderr)
        sys.exit(-1)
    if not os.path.isfile(args.input_dataset_file):
        print(f"Error: '{args.input_dataset_file}' is not a file", file=sys.stderr)
        sys.exit(-1)
        pass

    if not os.path.isdir(args.output_dataset_path):
        try:
            os.makedirs(os.path.basename(args.output_dataset_path))
        except: pass
        pass
    pass


def main(args):
    
    check_cmd_line_args(args)

    models_df = pd.read_csv(args.input_dataset_file)
    time_stamp, res_csv_path, res_test_df = run_tests_in_batch(args, models_df)

    merge_datasets(models_df, res_test_df, time_stamp, args)
    pass


if __name__ == "__main__":
    try:
        parser = get_custom_parser_test_batch()
        args = parser.parse_args()
        main(args)
    except:
        print(f"{str(err)}")
    finally:
        print(f"Done.")
        pass
    pass
