from dataclasses import dataclass

@dataclass
class LatentHyperparams:
    bs_mean: float
    bs_std: float
    
    bq_mean: float
    bq_std: float

    xs_mean: float
    xs_std: float
    
    xq_mean: float
    xq_std: float
