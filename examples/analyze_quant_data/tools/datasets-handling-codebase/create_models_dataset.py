from src.libraries.all_libs import *
from src.generics.custom_cmd_line_parsers.evaluate_models_custom_argparse import get_evaluate_models_argparser
from src.generics.utils import read_conf_file_content
from src.generics.utils import traverse_directory, get_overall_stats_from_input_dirs

from src.create_datasets.create_dataset_trains import create_dataset_baseline_trains
from src.create_datasets.create_dataset_trains import create_dataset_pruned_models_trains
from src.create_datasets.create_dataset_trains import create_enanched_datasets_train


def main(args) -> None:

    conf_dict = read_conf_file_content(args.conf_file)
    # pprint(conf_dict)

    if args.summary_estimated_workload:
        get_overall_stats_from_input_dirs(conf_dict, 1)
        return
        
    # get_overall_stats_from_input_dirs(conf_dict)
    kind_dataset = conf_dict["kind_dataset"]
    if conf_dict["kind_dataset"] == "baseline":
        # raise NotImplementedError(f"Chioce: '{kind_dataset}' not yet implemented!")
        dest_file_path_csv, timestamp_file, data_trains_df = \
            create_dataset_baseline_trains(conf_dict)
    elif conf_dict["kind_dataset"] == "pruned":
        dest_file_path_csv, timestamp_file, data_trains_df = \
            create_dataset_pruned_models_trains(conf_dict)
    elif conf_dict["kind_dataset"] == "quanted":
        raise NotImplementedError(f"Chioce: '{kind_dataset}' not yet implemented!")
    else:
        raise NotImplementedError(f"Chioce: '{kind_dataset}' not allowed!")
    
    if timestamp_file is None:
        print("No processing has been done since data location provided containes not useful resources to create an output dataset.")
        sys.exit(0)
        pass
    if conf_dict['create_enanched_datasets']:
        dest_file_path_csv_2, _ = create_enanched_datasets_train(data_trains_df, conf_dict, timestamp_file)
        pass
    
    meta_data = dict(
        tabular_data = [
            ['dest_file_path_csv', dest_file_path_csv],
            ['dest_file_path_csv_2', dest_file_path_csv_2],
        ]
    )
    table = tabulate.tabulate(**meta_data)
    print(table)
    pass


if __name__ == "__main__":
    parser = get_evaluate_models_argparser()
    args = parser.parse_args()
    main(args)
    pass
