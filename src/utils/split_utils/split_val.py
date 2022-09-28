import torch
from src.config.SplitParams import SplitParams

def split_val(train_ts, split_params: SplitParams):
    test_split, val_split = split_params.test_split, split_params.val_split
    val_of_train_split = val_split/(1 - test_split)

    num_val_entries = int(train_ts.shape[1] * val_of_train_split)
    num_train_entries = train_ts.shape[1] - num_val_entries
    train_ts, val_ts = torch.split(train_ts, [num_train_entries, num_val_entries], dim=1)
    return train_ts, val_ts
