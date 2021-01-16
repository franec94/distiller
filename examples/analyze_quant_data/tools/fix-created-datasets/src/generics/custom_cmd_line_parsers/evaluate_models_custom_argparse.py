from src.libraries.std_libs import *

def get_evaluate_models_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf_file", dest='conf_file', type=str, required=True, \
        help="Input file path, within local file system to yaml or json config file with constraints or specifications to carry out models' evalute task."
    )
    return parser