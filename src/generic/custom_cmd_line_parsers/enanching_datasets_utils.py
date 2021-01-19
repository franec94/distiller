from src.libraries.std_libs import *
from src.libraries.data_science_libs import *
from src.generics.utils import read_conf_file_content

def get_custom_calculate_prune_class(key_index_dict: dict):
    """
    Return custom function for calculating prune class.
    Args:
    -----
    `key_index_dict` - dict object with relation between columns names and index position within a series object retrieved from a dataframe instance.\n
    Returns:
    --------
    `custom function` - custom function with 'key_index_dict' embedded within returned function.
    """
    def calculate_prune_class(item, key_index_dict=key_index_dict):
        classes_set = set()
        scheduler_pos = key_index_dict['scheduler']
        if scheduler_pos == "" or \
            scheduler_pos == "-":
            return "-"
        if type(item[scheduler_pos]) == str:
            if item[scheduler_pos] != "" and item[scheduler_pos] != "-":
                scheduler_dict = eval(item[scheduler_pos])
            else:
                return item[scheduler_pos]
        else:
            scheduler_dict = item[scheduler_pos]
        for k, v in scheduler_dict['pruners'].items():
            classes_set.add(v['class'])
            pass
        map_to_shorter_class_name = lambda item: ''.join(list(filter(lambda char: char.lower() != char, item)))
        classes_list = list(map(map_to_shorter_class_name, list(classes_set)))
        classes_prunining = '+'.join(classes_list)
        return classes_prunining
    return calculate_prune_class


def get_custom_calculate_prune_rate(key_index_dict: dict):
    """
    Return custom function for calculating prune rate adopted.
    Args:
    -----
    `key_index_dict` - dict object with relation between columns names and index position within a series object retrieved from a dataframe instance.\n
    Returns:
    --------
    `custom function` - custom function with 'key_index_dict' embedded within returned function.
    """
    def calculate_prune_rate(a_row, key_index_dict=key_index_dict):
        # Raw data employed.
        if a_row[key_index_dict["prune_rate"]] != 0.: return a_row[key_index_dict["prune_rate"]]
        n_hf, n_hl = int(a_row[key_index_dict["n_hf"]]), int(a_row[key_index_dict["n_hl"]])
        scheduler_pos = key_index_dict["scheduler"]
        if type(a_row[scheduler_pos]) == str:
            if a_row[scheduler_pos] != "" and a_row[scheduler_pos] != "-":
                scheduler_dict = eval(a_row[scheduler_pos])
            else:
                return 0.0
        else:
            scheduler_dict = a_row[scheduler_pos]
        # print(n_hf, n_hl)
        
        # Models's derived infos.
        biases_list = np.array([2] + [n_hf] * n_hl + [1])
        wgts_list = np.array([n_hf*2] + [n_hf*n_hf] * n_hl + [n_hf])
        wgts_rates = np.ones(len(wgts_list))
        # print(len(biases_list), len(wgts_list), len(wgts_rates))
        
        for k, v in scheduler_dict['pruners'].items():
            final_sparsity = v['final_sparsity']
            for a_wgt in v['weights']:
                pos = int(a_wgt.split(".")[2])
                # print(a_wgt, pos)
                wgts_rates[pos] = final_sparsity
                pass
            prune_rate = (sum(wgts_rates * wgts_list) + sum(biases_list)) / (sum(wgts_list) + sum(biases_list))
            pass
        # return np.round(prune_rate, 2)
        return prune_rate
    return calculate_prune_rate


def get_custom_cmd_line_opts_dict(key_index_dict: dict):
    """
    Return custom function for retrieving output dictionary referring to options adopted at training time to train each model.
    Args:
    -----
    `key_index_dict` - dict object with relation between columns names and index position within a series object retrieved from a dataframe instance.\n
    Returns:
    --------
    `custom function` - custom function with 'key_index_dict' embedded within returned function.
    """
    def get_cmd_line_opts_dict(item, key_index_dict=key_index_dict):
        targets = "sidelength,n_hf,n_hl,num_epochs,lr,lambda_L_1,lambda_L_2".split(",")
        default_targets = [256,0,0,1e+3,1e-3,.0,.0]
        targets_dict = dict(zip(targets, default_targets))
        command_line = item[key_index_dict['command_line']]
        a_row_list = command_line.split("--")
        for a_opt in a_row_list:
            for a_target in targets:
                if a_opt.startswith(a_target):
                    targets_dict[a_target] = ' '.join(a_opt.split(" ")[1:])
                    pass
                pass
            pass
        return targets_dict
    return get_cmd_line_opts_dict
