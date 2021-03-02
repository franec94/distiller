from src_cpd.libs.std_libs import *
from src_cpd.libs.project_libs import *


def create_out_dir(args, a_run_ts) -> None:
    """Create out directory where results will be stored.
    Args:
    -----
    `args` - namespace object. Default None, if None return without doing anything.\n
    `a_run_ts` - time stamp object.\n
    """

    args.out_dir = os.path.join(args.out_dir, f"res_out_{str(a_run_ts)}")
    try:
        os.makedirs(args.out_dir)
        a_filename: str = os.path.basename(args.conf_filepath)
        copy_conf_fp: str = os.path.join(args.out_dir, a_filename)
        shutil.copy(args.conf_filepath, copy_conf_fp)
    except Exception as err:
        raise err

    pass


def check_command_line_args(args) -> None:
    """Check Command line args.
    Args:
    -----
    `args` - namespace object. Default None, if None return without doing anything.
    """

    check_file_exists(a_filepath=args.conf_filepath)
    check_dir_exists(a_dirpath=args.root_dir)
    pass


def main(args=None) -> None:
    """Main function that orchestrates whatever job such a program is able to handle.
    Args:
    -----
    `args` - namespace object. Default None, if None return without doing anything.\n
    """
    if args is None: return

    # Setup script for properly accomplishing and fullfilling target jobs or tasks.
    a_run_ts = time.time()
    check_command_line_args(args=args)
    create_out_dir(args=args, a_run_ts=a_run_ts)

    conf_data: dict = read_conf_file(args.conf_filepath, raise_exception=False)

    # Create output dataset
    a_df = create_out_dataset(args=args, conf_data=conf_data)
    
    # run_tests_in_batch(args=args, models_df=a_df, verbose = 0)
    merged_df = merge_performace_w_models_data(args=args, models_df=a_df)

    a_df_path = os.path.join(args.out_dir, f"merged_out.csv")
    merged_df.to_csv(a_df_path)
    pass


if __name__ == "__main__":
    """Entry Point when such a python source file is runned as stand alone script or program from command line."""

    parser = get_custom_argparser()
    args = parser.parse_args()

    main(args=args)


    try:
        # main()
        pass
    except Exception as err:
        print(f"{str(err)}")
        pass
    pass
