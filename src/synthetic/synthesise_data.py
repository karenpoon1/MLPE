import os
import torch

from src.models.M2PL import M2PL
from src.synthetic.latent_config import latent_config_dict

# Define variables
model_dimension = 1
data_dim = [500, 20]
latent_config = 'int_latent_params_1'
latent_params = latent_config_dict[latent_config]
random_state = 1200
path_to_folder = f'data/synthetic/Model{model_dimension}D2PL_S{data_dim[0]}Q{data_dim[1]}_LatentConfig{latent_config}_Random{random_state}/'

# Synthesise data
my_model = M2PL(dimension=model_dimension)
data_df, true_latents = my_model.synthesise_data(data_dim, latent_params, random_state)

# Save data
synthetic_data = {'data_df': data_df, 'true_latents': true_latents}
if not os.path.exists(path_to_folder):
    os.makedirs(path_to_folder)
torch.save(synthetic_data, path_to_folder + 'data.pt')

print(data_df)
print(true_latents)
