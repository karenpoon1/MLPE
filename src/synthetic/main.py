import torch

from src.config.split_config import split_config_dict
from src.config.model_config import model_config_dict

from src.synthetic.SyntheticParams import SyntheticParams

from src.utils.split_utils.split_train_test import split_train_test
from src.utils.split_utils.split_val import split_val

from src.models.M2PL import M2PL
from src.eval_model import eval_M2PL_model

from src.utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix

# Load synthetic data
synthetic_params = SyntheticParams(model_dim=1, data_dim=[5000, 200],
                        latent_config='int_latent_params_8', random_state=1200)

synthetic_data = torch.load(synthetic_params.get_data_path() + 'data.pt')
data_df, true_latents = synthetic_data['data_df'], synthetic_data['true_latents']


# Define run variables
split_config = 'default'
split_random_state = 1300

model_dim = 0
model_config = 'default'
init_random_state = 1300

results_folder_name = f'SplitConfig_{split_config}__Random{split_random_state}/Model{model_dim}D2PL__ModelConfig_{model_config}__InitRandom{init_random_state}'

# Get params
split_params = split_config_dict[split_config]
hyperparams = model_config_dict[model_config]

# split to train test set
train_ts, test_ts = split_train_test(data_df,
                                        split_params,
                                        split_random_state)

# split val set
train_ts, val_ts = split_val(train_ts,
                                split_params)


# eval model
my_model = M2PL(dim=model_dim)
data_dim = data_df.shape


# Prediction with true latents
probit_correct, thres_predictions_ts = my_model.predict(test_ts, true_latents)
acc = calc_acc(test_ts[0], thres_predictions_ts)
conf_matrix = calc_conf_matrix(test_ts[0], thres_predictions_ts)
print(acc)
print(conf_matrix)


print('\nafter training\n')


# fit model
data_folder = synthetic_params.get_data_path()
results_path = f'{data_folder}/results/{results_folder_name}/'
res = my_model.run(train_ts, test_ts, val_ts, data_dim,
                        hyperparams, init_random_state,
                        plot=True, save=results_path)
                        
print(res['results']['performance']['acc'])
