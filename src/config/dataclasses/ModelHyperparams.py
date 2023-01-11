from dataclasses import dataclass

from typing import Literal
from src.config.dataclasses.LatentHyperparams import LatentHyperparams
from src.config.latent_hyperparams_config import latent_config_dict

@dataclass
class ModelHyperparams:
    rate: float = 0.00015
    iters: int = 10000
    stop_method: Literal['nll', 'acc', ''] = 'nll'
    latent_hyperparams: LatentHyperparams = latent_config_dict['default']
