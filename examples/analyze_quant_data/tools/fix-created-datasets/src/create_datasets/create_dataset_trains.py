from src.libraries.std_libs import *
from src.libraries.data_science_libs import *
from src.generics.utils import read_conf_file_content

def create_dataset_trains(conf_dict):
    field_names = "date_train,date_test,model_name,psnr,bpp,CR,mse,ssim,scheduler_name,scheduler,agp_tech,quant_tech,command_line".split(",")
    TrainRecord = collections.namedtuple("TrainRecord", field_names)

    root_trains_list = []
    for a_dir_path in conf_dict['input_dirs_list']:
        for dir_name, subdirs_list, files_list in os.walk(a_dir_path):
            if "configs" not in subdirs_list: continue

            models_files_list = list(filter(lambda item: item.endswith(".tar"), files_list))
            train_logs_list = list(filter(lambda item: item.endswith(".log"), files_list))[0]
            date_train = os.path.basename(dir_name).replace("_", "")

            with open(os.path.join(dir_name, train_logs_list)) as log_fp:
                lines = log_fp.read().split("\n")
                command_line = list(filter(lambda item: "Command line" in item, lines))[0]
                pass

            data_record = ["-","-","-",0,0,0,0,0,"-","-","-","-","-"]
            data_record[12] = command_line
            assert len(data_record) == len(field_names)

            configs_subdir = os.path.join(dir_name, "configs")
            scheduler = None
            for dir_name, subdirs_list, files_list in os.walk(configs_subdir):
                scheduler = files_list[0]
                pass
            for ii, a_model in enumerate(models_files_list):
                data_record[0] = date_train if ii == 0 else f"{date_train}-no_{ii}"
                data_record[1] = "-"
                data_record[2] = a_model
                

                if scheduler:
                    data_record[8] = scheduler
                    scheduler_path = os.path.join(configs_subdir, scheduler)
                    data_record[9] = read_conf_file_content(scheduler_path)
                    pass

                a_record = TrainRecord._make(data_record)
                root_trains_list.append(a_record)
            pass
        pass
    if root_trains_list != []:
        data = list(map(lambda item: item._asdict(), root_trains_list))
        data_trains_df = pd.DataFrame(data = data, columns = field_names)
        print(data_trains_df.head(5))

        print(data_trains_df.info())
        print(data_trains_df["command_line"].head(5))

        timestamp_file = time.time()
        dest_dir_csv = os.path.join(conf_dict["root_dest_results"], f"pool_{timestamp_file}")
        dest_file_path_csv = os.path.join(dest_dir_csv, f"pool_{timestamp_file}.csv")

        if not os.path.isdir(dest_dir_csv):
            try:
                os.makedirs(dest_dir_csv)
                print(f"Dest directory '{dest_dir_csv}' created!")
                pass
            except:
                print(f"Dest directory '{dest_dir_csv}' already exists!")
                pass
        else: 
            print(f"Dest directory '{dest_dir_csv}' already exists!")
            pass
        data_trains_df.to_csv(dest_file_path_csv)
        pass
    pass