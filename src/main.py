import configparser

from src.utils.data_utils.parse_paper import parse_paper
from src.utils.data_utils.process_paper import process_paper

from src.utils.split_utils.split_train_test import split_train_test
from src.utils.split_utils.split_val import split_val

from src.models.M2PL import M2PL

from src.config.split_config import split_config_dict
from src.config.model_config import model_config_dict

# Define variables
dataset = 'b1'
selected_papers = [0]
split_random_state = 1000
split_config = 'default'
split_params = split_config_dict[split_config]

model_dimension = 0
model_config = 'default'
hyperparams = model_config_dict[model_config]
stop_method = 'nll'
init_random_state = 1000

path_to_folder = (f'results/dataset_{dataset}/paper{selected_papers[0]}-{selected_papers[-1]}_splitconfig{split_config}_random{split_random_state}\n'
                    f'/model{model_dimension}D2PL_modelconfig_{model_config}_stop{stop_method}_initRandom{init_random_state}/')

# parse paper
data_dfs, meta_dfs = parse_paper(dataset)

# process paper
processed_data_df, processed_meta_df = process_paper(data_dfs, meta_dfs, selected_papers=selected_papers)
data_dim = processed_data_df.shape

# split to train test set
train_ts, test_ts = split_train_test(processed_data_df,
                                        split_params,
                                        random_state=split_random_state)

# split val set
train_ts, val_ts = split_val(train_ts,
                                split_params)

# run model
my_model = M2PL(dimension=model_dimension)
info = my_model.run(train_ts, test_ts, val_ts, data_dim, hyperparams,
                        stop_method=stop_method, init_random_state=init_random_state,
                        plot=False, save=path_to_folder)
# print(info)
print(info['results']['performance']['acc'])
