from typing import List
from dataclasses import dataclass

@dataclass
class SyntheticParams:
    model_dim: int
    data_dim: List[int]
    latent_config: str
    random_state: int
    
    def get_folder_name(self): # def __post_init__(self)
        return f'Model{self.model_dim}D2PL__S{self.data_dim[0]}Q{self.data_dim[1]}__LatentConfig_{self.latent_config}__Random{self.random_state}'

    def get_data_path(self):
        return f'src/synthetic/data/{self.get_folder_name()}/'

