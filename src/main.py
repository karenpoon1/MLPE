from src.utils.data_utils.parse_paper import parse_paper
from src.utils.data_utils.process_paper import process_paper

from src.config.split_config import split_config_dict
from src.config.model_config import model_config_dict

from src.eval_model import eval_M2PL_model

# Load data
dataset = 'b1'
selected_papers = [0]
data_dfs, meta_dfs = parse_paper(dataset) # parse paper
data_df, meta_df = process_paper(data_dfs, meta_dfs, selected_papers=selected_papers) # process paper


# Define variables
split_config = 'default'
split_random_state = 1000

model_dim = 0
model_config = 'test'
init_random_state = 1000

# Folder name to save results
path_to_folder = (f'results/dataset_{dataset}/Paper{selected_papers[0]}-{selected_papers[-1]}/SplitConfig_{split_config}__Random{split_random_state}/Model{model_dim}D2PL__ModelConfig_{model_config}__InitRandom{init_random_state}/')

# Evaluate model
split_params = split_config_dict[split_config]
hyperparams = model_config_dict[model_config]
res = eval_M2PL_model(model_dim, data_df, split_params, split_random_state, 
                hyperparams, init_random_state, plot=False, save=path_to_folder)

print(res['results']['performance']['acc'])
