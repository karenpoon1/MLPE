from src.config.split_config import split_config_dict
from src.config.model_config import model_config_dict

from src.utils.data_utils.parse_paper import parse_paper
from src.utils.data_utils.process_paper import process_paper

from src.utils.split_utils.split_train_test_val import split_train_test, split_val_from_train


def eval_model(dataset, selected_papers,
         split_config, split_seed,
         model, model_config, init_seed,
         plot, save, step_size=25):

    data_df, meta_df = load_data(dataset, selected_papers)
    train_ts, test_ts, val_ts = split_data(data_df, split_config, split_seed)
    return fit_model(train_ts, test_ts, val_ts, data_df.shape,
                    model, model_config, init_seed,
                    plot, save, step_size)


def load_data(dataset, selected_papers):
    data_dfs, meta_dfs = parse_paper(dataset) # parse paper
    data_df, meta_df = process_paper(data_dfs, meta_dfs, selected_papers=selected_papers) # process paper
    return data_df, meta_df


def split_data(data_df, split_config, split_seed):
    split_params = split_config_dict[split_config]

    train_ts, test_ts = split_train_test(data_df, split_params.test_split, split_seed)
    train_ts, val_ts = split_val_from_train(train_ts, split_params.val_of_train_split)
    return train_ts, test_ts, val_ts


def fit_model(train_ts, test_ts, val_ts, data_dim,
               model, model_config, init_seed,
               plot=False, save=False, step_size=25):
    
    hyperparams = model_config_dict[model_config]
    return model.run(train_ts, test_ts, val_ts, data_dim,
                     hyperparams, init_seed,
                     plot, save, step_size)
