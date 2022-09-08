from src.utils.data_utils.parse_paper_b01 import parse_paper_b1
from src.utils.data_utils.parse_paper_b2 import parse_paper_b2
from src.utils.data_utils.process_paper import process_paper

from src.split_train_test import split_train_test
from src.split_val import split_val

from src.split_config import default as split_params


# parse paper
data_dfs, meta_dfs = parse_paper_b2()

# process paper
processed_data_df, processed_meta_df = process_paper(data_dfs, meta_dfs, selected_papers=[0,1])
print(processed_data_df)
print(processed_meta_df)

# split to train test set
train_ts, test_ts = split_train_test(processed_data_df,
                                        split_params,
                                        random_state=1000)

# split val set
train_ts, val_ts = split_val(train_ts,
                                split_params['val_split'])
