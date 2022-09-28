from dataclasses import dataclass

@dataclass
class SplitParams:
    random_q_order: bool = True
    random_s_order: bool = False
    test_split: float = 0.2
    val_split: float = 0.1
