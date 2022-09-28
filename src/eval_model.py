from src.config.SplitParams import SplitParams
from src.config.ModelParams import ModelParams

from src.utils.split_utils.split_train_test import split_train_test
from src.utils.split_utils.split_val import split_val

from src.models.M2PL import M2PL

def eval_M2PL_model(model_dim: int, data_df, split_params: SplitParams, split_random_state: int,
                    hyperparams: ModelParams, init_random_state: int,
                    plot=False, save=False, step_size=25):

    # split to train test set
    train_ts, test_ts = split_train_test(data_df,
                                            split_params,
                                            split_random_state)

    # split val set
    train_ts, val_ts = split_val(train_ts,
                                    split_params)

    # run model
    my_model = M2PL(dim=model_dim)
    data_dim = data_df.shape
    res = my_model.run(train_ts, test_ts, val_ts, data_dim,
                            hyperparams, init_random_state,
                            plot, save, step_size)

    return res
