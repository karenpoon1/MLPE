import os
import torch

from src.models.M2PL import M2PL
from src.config.latent_config import latent_config_dict

from src.synthetic.SyntheticParams import SyntheticParams

# Define variables
synthetic_params = SyntheticParams(model_dim=1, data_dim=[5000, 200],
                        latent_config='int_latent_params_8', random_state=1200)

# Synthesise data
my_model = M2PL(dim=synthetic_params.model_dim)
latent_params = latent_config_dict[synthetic_params.latent_config]
data_df, true_latents = my_model.synthesise_data(synthetic_params.data_dim, latent_params,
                                                    synthetic_params.random_state)

# Save data
path_to_folder = synthetic_params.get_data_path()

synthetic_data = {'data_df': data_df, 'true_latents': true_latents}
if not os.path.exists(path_to_folder):
    os.makedirs(path_to_folder)
torch.save(synthetic_data, path_to_folder + 'data.pt')
