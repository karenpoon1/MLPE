import torch


def split_trailing_entries(data_ts: torch.Tensor, trailing_split: float):
    # Split the last 'e.g. trailing_split=30%' out as trailing set, the rest as leading set
    num_trailing_entries = int(data_ts.shape[1] * trailing_split)
    num_leading_entries = data_ts.shape[1] - num_trailing_entries
    leading_ts, trailing_ts = torch.split(data_ts, [num_leading_entries, num_trailing_entries], dim=1)
    return leading_ts, trailing_ts


def split_chunks(data_ts, n_chunks):
    chunk_size = int(data_ts.shape[1]/n_chunks) # round down
    train_chunks = torch.split(data_ts, chunk_size, dim=1)
    return train_chunks
