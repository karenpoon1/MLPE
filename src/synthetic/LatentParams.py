from dataclasses import dataclass

@dataclass
class LatentParams:
    bs_mean: float
    bs_std: float
    
    bq_mean: float
    bq_std: float

    xs_mean: int
    xs_std: float
    
    xq_mean: float
    xq_std: float
