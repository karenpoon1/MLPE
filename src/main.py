from src.utils.data_utils.parse_paper import parse_paper
from src.utils.data_utils.process_paper import process_paper

from src.utils.split_utils.split_train_test import split_train_test
from src.utils.split_utils.split_val import split_val

from src.models.M2PL import M2PL

from src.config.split_config import default as split_params
from src.config.model_config import default as hyperparams

# Define variables
model_dimension = 0
dataset = 'b1'
selected_papers = [0]
random_state = 1000
stop_method = 'nll'
init_random_state = 1000
save = f'results/paper_{dataset}/model{model_dimension}D2PL_paper{selected_papers[0]}-{selected_papers[-1]}_random{random_state}_stop{stop_method}_initRandom{init_random_state}/'

# parse paper
data_dfs, meta_dfs = parse_paper(dataset)

# process paper
processed_data_df, processed_meta_df = process_paper(data_dfs, meta_dfs, selected_papers=selected_papers)
params_dim = processed_data_df.shape

# split to train test set
train_ts, test_ts = split_train_test(processed_data_df,
                                        split_params,
                                        random_state=random_state)

# split val set
train_ts, val_ts = split_val(train_ts,
                                split_params)

# run model
my_model = M2PL(dimension=model_dimension)
info = my_model.run(train_ts, test_ts, val_ts, params_dim, hyperparams,
                        stop_method=stop_method, init_random_state=init_random_state,
                        plot=True, save=save)
# print(info)
print(info['results']['performance']['acc'])
