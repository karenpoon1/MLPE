from src.config.LatentHyperparams import LatentHyperparams

latent_config_dict = {
    'AD_latent_params_1': LatentHyperparams(0, 1, 0, 1, float('nan'), float('nan'), float('nan'), float('nan')),
    
    'AD_latent_params_2': LatentHyperparams(0, 1, -3, 1, float('nan'), float('nan'), float('nan'), float('nan')),

    'int_latent_params_1': LatentHyperparams(0, 1, 0, 1, 0, 1, 0, 1),

    'int_latent_params_2': LatentHyperparams(0, 1, 0, 1, 0, 3, 0, 3),

    'int_latent_params_3': LatentHyperparams(0, 3, 0, 3, 0, 3, 0, 3),

    'int_latent_params_4': LatentHyperparams(0, 3, 0, 3, 0, 5, 0, 5),

    'int_latent_params_5': LatentHyperparams(0, 1, -3, 1, 0, 0.1, 0, 0.1),

    'int_latent_params_6': LatentHyperparams(0, 1, -3, 1, 0, 1, 0, 1),

    'int_latent_params_7': LatentHyperparams(0, 1, 0, 1, 0, 0.5, 0, 0.5),

    'int_latent_params_8': LatentHyperparams(0, 1, 0, 1, 0, 0.1, 0, 0.1)

}
