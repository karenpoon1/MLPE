# MLPE

## Install required packages
    pip3 install -r requirements.txt


## Unit tests
#### Find all tests:
    python3 -m unittest discover tests -v

## Data
    data_dfs, meta_dfs = parse_paper_{dataset}

`dataset`:

`b1`: 2017 GCSE 1H 2H 3H dataset

`b2`: large dataset

    processed_data_df, processed_meta_df = process_paper(data_dfs, meta_dfs, {selected_papers})

`selected_papers` is a list of integers representing exam papers selected to be processed, e.g. `[0, 1, 2]` represents exam paper 0, paper 1, and paper 2 are selected to be the input data for modelling

## Split to train, test and validation set
    train_ts, test_ts = split_train_test(processed_data_df,
                                            split_params,
                                            {random_state})

    train_ts, val_ts = split_val(train_ts,
                                    split_params)

`split_params` is a dictionary of parameters imported from and defined in `src/config/split_config.py`

`random_state` is a random seed integer that can be changed
