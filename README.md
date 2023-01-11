# MLPE

## Python version
Developed under Python version 3.9.14 with pyenv

## Install required packages
    pip install -r requirements.txt


## Unit tests
#### Run all tests:
    python -m unittest discover tests -v
#### Run specific test:
    python -m unittest tests.utils.test_

## Data
    data_df, meta_df = load_data(<dataset>, <selected_papers>)

`dataset`:

- `b0`: 2017 GCSE 1H dataset

- `b1`: 2017 GCSE 1H 2H 3H dataset

- `b2`: large dataset

`selected_papers`:

A list of integers representing exam papers selected to be processed, e.g. `[0, 1, 2]` represents exam paper 0, paper 1, and paper 2 are selected to be the input data for modelling

## Split to train, test and validation set
    train_ts, test_ts = split_train_test(processed_data_df,
                                            split_params,
                                            {random_state})

    train_ts, val_ts = split_val(train_ts,
                                    split_params)

`split_params` is a dictionary of parameters imported from and defined in `src/config/split_config.py`

`random_state` is a random seed integer that can be changed

## Synthetic Run
Variables for generating synthetic data:

`DATA_DIM`: [student_dimension, question_dimension]

`LATENT_HYPERPARAMS_CONFIG`: config of hyperparameters (mean, std) of latent parameters defined in src/config/latent_hyperparams_config.py

`INIT_SEED`: random seed for generating latent parameters

`SYNTH_SEED`: random seed for generating data from latent parameters

`MODEL_DIM`: dimension of model which synthesised the data (0 - Ability-Difficulty; 1 - 1D Interactive, etc)

Synthetic data is stored as *data.pt* in the directory:

`data_dir` = "src/synthetic/data/Model`<MODEL_DIM>`D2PL__S`<student_dimension>`Q`<question_dimension>`__LatentConfig_`<LATENT_HYPERPARAMS_CONFIG>`__InitSeed`<INIT_SEED>`__SynthSeed`<SYNTH_SEED>`"

### Given synthetic data, a model is trained to fit the data with the following configuration settings

`SPLIT_CONFIG`: config of splitting data into train, test, and validation set defined in src/config/split_config.py, including train test split

`SPLIT_SEED`: random seed to shuffle data before splitting into train and test set

`MODEL_DIM`: dimension of model to fit the synthetic data

`MODEL_CONFIG`: config of hyperparameters of model, including training rate, iterations, hyperparameters of latents, and method of stop training

`INIT_SEED`: random seed for initialising latent parameters to be trained
