import pandas as pd
from typing import List, Tuple

from src.utils.df_utils.combine_df import combine_df
from src.utils.df_utils.threshold_df import threshold_df
from src.utils.df_utils.binarise_df import binarise_df
from src.utils.df_utils.reset_index_df import reset_col_index, reset_row_index

def process_paper(data_dfs, meta_dfs, selected_papers: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Slice out selected papers
    selected_data_dfs = [data_dfs[i] for i in selected_papers]
    selected_meta_dfs = [meta_dfs[i] for i in selected_papers]

    # Combine papers
    combined_data_df = combine_df(selected_data_dfs)
    combined_meta_df = combine_df(selected_meta_dfs)

    # Threshold and binarise data
    max_scores_df = combined_meta_df.loc['Max'].astype(float)
    processed_data_df = threshold_df(combined_data_df, max_scores_df)
    processed_data_df = binarise_df(processed_data_df, max_scores_df)

    # Reset index from 0 to last
    processed_data_df = reset_row_index(processed_data_df)
    processed_data_df = reset_col_index(processed_data_df)
    processed_meta_df = reset_col_index(combined_meta_df)
    
    return processed_data_df, processed_meta_df
