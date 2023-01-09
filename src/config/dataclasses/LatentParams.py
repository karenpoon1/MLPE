from dataclasses import dataclass, asdict, field
import torch

@dataclass
class LatentParams:
    bs0: torch.Tensor
    bq0: torch.Tensor
    xs: torch.Tensor = torch.Tensor([])
    xq: torch.Tensor = torch.Tensor([])

    bs: torch.Tensor = field(init=False)
    bq: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.bs = torch.concat([self.bs0, self.xs], dim=1)
        self.bq = torch.concat([self.bq0, self.xq], dim=1)

    @classmethod
    def return_obj(cls, latents_dict):
        bs, bq = latents_dict['bs'].detach(), latents_dict['bq'].detach()
        bs0, xs  = bs[:, 0:1], bs[:, 1:]
        bq0, xq  = bq[:, 0:1], bq[:, 1:]
        return cls(bs0, bq0, xs, xq)

    def as_dict(self):
        return asdict(self)

    def get_simplified_dict(self):
        return {'bs': self.bs, 'bq': self.bq}
