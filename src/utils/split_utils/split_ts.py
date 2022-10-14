import torch


def split_ts(data_ts, split):
    num_val_entries = int(data_ts.shape[1] * split)
    num_train_entries = data_ts.shape[1] - num_val_entries
    selected_ts, remaining_ts = torch.split(data_ts, [num_train_entries, num_val_entries], dim=1)
    return selected_ts, remaining_ts


def split_chunks(data_ts, n_chunks):
    chunk_size = int(data_ts.shape[1]/n_chunks) # round down
    train_chunks = torch.split(data_ts, chunk_size, dim=1)
    return train_chunks
