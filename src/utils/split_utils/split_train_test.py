import torch
from src.config.SplitParams import SplitParams

from src.utils.df_utils.shuffle_df import shuffle_cols, shuffle_rows
from src.utils.df_utils.vectorise_df import vectorise_df
from src.utils.ts_utils.shuffle_ts import shuffle_2D_vect_ts


def split_train_test(data_df, split_params: SplitParams, random_state=1000):
    if split_params.random_q_order:
        data_df = shuffle_cols(data_df, shuffle_seed=random_state, reset_index=False) # shuffle question order
    
    if split_params.random_s_order:
        data_df = shuffle_rows(data_df, shuffle_seed=random_state, reset_index=False) # shuffle student order
    
    # Vectorise and shuffle selected_df
    vectorised_ts = vectorise_df(data_df)
    vectorised_ts = shuffle_2D_vect_ts(vectorised_ts, shuffle_seed=random_state)

    # Split the last 'e.g. test_split=30%' out as test set, the rest as train set
    num_test_entries = int(vectorised_ts.shape[1] * split_params.test_split)
    num_train_entries = vectorised_ts.shape[1] - num_test_entries
    train_ts, test_ts = torch.split(vectorised_ts, [num_train_entries, num_test_entries], dim=1)

    return train_ts, test_ts
