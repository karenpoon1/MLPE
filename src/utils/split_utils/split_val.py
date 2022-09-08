import torch

def split_val(train_ts, val_split):
    num_val_entries = int(train_ts.shape[1] * val_split)
    num_train_entries = train_ts.shape[1] - num_val_entries
    train_ts, val_ts = torch.split(train_ts, [num_train_entries, num_val_entries], dim=1)
    return train_ts, val_ts
