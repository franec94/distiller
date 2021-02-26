from src.libs.std_python_libs import *
from src.libs.project_libs import *


def create_out_dir(args, a_run_ts, raise_exception:bool = True) -> None:
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
        if raise_exception:
            raise err
        print(f"Exception: {str(err)} managed.")
        pass

    pass


def check_command_line_args(args) -> None:
    """Check Command line args.
    Args:
    -----
    `args` - namespace object. Default None, if None return without doing anything.
    """

    functions.check_file_exists(file_path=args.conf_filepath)
    functions.check_dir_exists(dir_path=args.root_dir)
    pass

