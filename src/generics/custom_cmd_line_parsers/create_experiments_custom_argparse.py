from src.libraries.std_libs import *

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf_file", dest='conf_file', type=str, required=True, \
        help="Input file path, within local file system to yaml or json config file with constraints or specifications to carry out models' evalute task."
    )
    parser.add_argument("--summary_estimated_workload", dest='summary_estimated_workload', action="store_true", required=False, default=False, \
        help="Show estimated workload details, then end script."
    )
    return parser