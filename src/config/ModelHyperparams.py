from dataclasses import dataclass

from typing import Literal
from src.config.LatentHyperparams import LatentHyperparams

@dataclass
class ModelHyperparams:
    rate: float = 0.00015
    iters: int = 10000
    stop_method: Literal['nll', 'acc', ''] = 'nll'
    latent_hyperparams: LatentHyperparams = LatentHyperparams(0, 1, 0, 1, 0, 1, 0, 1)
