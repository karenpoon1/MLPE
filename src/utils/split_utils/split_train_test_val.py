from src.utils.df_utils.shuffle_df import shuffle_cols, shuffle_rows
from src.utils.df_utils.vectorise_df import vectorise_df
from src.utils.ts_utils.shuffle_ts import shuffle_2D_vect_ts
from src.utils.split_utils.split_ts import split_trailing_entries


def split_train_test(data_df, test_split, random_state=1000):
    # if split_params.random_q_order:
        # data_df = shuffle_cols(data_df, shuffle_seed=random_state, reset_index=False) # shuffle question order
    # if split_params.random_s_order:
        # data_df = shuffle_rows(data_df, shuffle_seed=random_state, reset_index=False) # shuffle student order

    # Vectorise and shuffle selected_df
    vectorised_ts = vectorise_df(data_df)
    vectorised_ts = shuffle_2D_vect_ts(vectorised_ts, shuffle_seed=random_state)

    # Split the last 'e.g. test_split=30%' out as test set, the rest as train set
    train_ts, test_ts = split_trailing_entries(vectorised_ts, test_split)
    return train_ts, test_ts


def split_val_from_train(train_ts, val_of_train_split):
    train_ts, val_ts = split_trailing_entries(train_ts, val_of_train_split)
    return train_ts, val_ts
