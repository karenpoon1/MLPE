# MLPE

## Install required packages
---
    pip3 install -r requirements.txt


## Unit tests
---
#### Find all tests:
    python3 -m unittest discover tests -v

## Data
---
    data_dfs, meta_dfs = parse_paper_{dataset}

`dataset`:

`b1`: 2017 GCSE 1H 2H 3H dataset

`b2`: large dataset

## Split to train and test set
---
    train_ts, test_ts = split_train_test(processed_data_df,
                                            split_params,
                                            random_state={random_seed})

`split_params` is a dictionary of parameters imported from and defined in `split_config.py`

`random_seed` can be changed

