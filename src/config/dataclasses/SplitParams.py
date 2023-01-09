from dataclasses import dataclass, field

@dataclass
class SplitParams:
    random_q_order: bool = True
    random_s_order: bool = False
    test_split: float = 0.2
    val_split: float = 0.1

    val_of_train_split: float = field(init=False)

    def __post_init__(self):
        # curr_train_frac * val_of_train_split = expected_val_frac
        # (1 - test_frac) * val_of_train_split = expected_val_frac
        # val_of_train_split = expected_val_frac / (1 - test_frac)
        self.val_of_train_split = self.val_split/(1 - self.test_split) 
