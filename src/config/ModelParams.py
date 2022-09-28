from dataclasses import dataclass

from typing import Literal
from src.config.LatentParams import LatentParams

@dataclass
class ModelParams:
    rate: float = 0.00015
    iters: int = 10000
    stop_method: Literal['nll', 'acc', ''] = 'nll'
    latent_params: LatentParams = LatentParams(0, 1, 0, 1, 0, 1, 0, 1)
