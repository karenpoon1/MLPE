import torch
from src.config.LatentParams import LatentParams

from src.models.M2PL import M2PL

from src.eval_model import split_data, fit_model
from src.utils.metric_utils.calc_metric import calc_acc
from src.config.latent_hyperparams_config import latent_config_dict

from src.print_info import print_info


# Variables for synthetic data
DATA_DIM = [5000, 200]
LATENT_HYPERPARAMS_CONFIG = 'int_latent_params_8'
INIT_SEED = 1200
SYNTH_SEED = 1200
MODEL_DIM = 1

synth_model = M2PL(MODEL_DIM)
data_folder = f'Model{MODEL_DIM}D2PL__S{DATA_DIM[0]}Q{DATA_DIM[1]}__LatentConfig_{LATENT_HYPERPARAMS_CONFIG}__InitSeed{INIT_SEED}__SynthSeed{SYNTH_SEED}'
data_dir = f'src/synthetic/data/{data_folder}/'

# Retrieve or generate synthetic data
try:
    # Retrieve
    synthetic_data = torch.load(data_dir + 'data.pt')
    synthetic_df, latents_dict = synthetic_data['data_df'], synthetic_data['latents']
    latents = LatentParams.return_obj(latents_dict)

    # Retrieve estimated latents
    # info = torch.load(target_dir + 'info.pt')
    # latents_dict = info['results']['params']
    # latents = LatentParams.return_obj(latents_dict)

except:
    # Generate
    latent_hyperparams = latent_config_dict[LATENT_HYPERPARAMS_CONFIG]
    synthetic_df, latents = synth_model.synthesise_from_hyperparams(DATA_DIM, latent_hyperparams, INIT_SEED, SYNTH_SEED, data_dir)


# Variables for evaluating model
SPLIT_CONFIG = 'default'
SPLIT_SEED = 1300

MODEL_DIM = 1
MODEL_CONFIG = 'default'
INIT_SEED = 1300

results_dir = f'{data_dir}results/SplitConfig_{SPLIT_CONFIG}__SplitSeed{SPLIT_SEED}/Model{MODEL_DIM}D2PL__ModelConfig_{MODEL_CONFIG}__InitSeed{INIT_SEED}/'

# Split data
train_ts, test_ts, val_ts = split_data(synthetic_df, SPLIT_CONFIG, SPLIT_SEED)

# Evaluate model
my_model = M2PL(MODEL_DIM)

# Predictions with ground truth latents
probit_correct, thres_predictions_ts = my_model.predict(test_ts, latents.get_simplified_dict())
acc = calc_acc(test_ts[0], thres_predictions_ts)
print(f'Predictions from ground truth latents - Accuracy: {acc}')

# Predictions after training
res = fit_model(train_ts, test_ts, val_ts, synthetic_df.shape,
                my_model, MODEL_CONFIG, INIT_SEED,
                plot=True, save=False)
acc = res['results']['performance']['acc']
print(f'After training - Accuracy: {acc}')
# print_info(results_dir)
