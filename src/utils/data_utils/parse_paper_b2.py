# Parse csv #2 only

import pandas as pd
from typing import List, Tuple

from src.utils.df_utils.select_rows_by_col_df import select_rows_by_col_df


# for file 'Large Pinpoint Learning Data Set for Research.csv' only
def parse_paper_b2():
    data_csv_name = "data/Large Pinpoint Learning Data Set for Research.csv"
    meta_csv_name = "data/Large Pinpoint Learning Data Set for Research meta.csv"

    raw_data_df = pd.read_csv(data_csv_name, low_memory=False)
    raw_meta_df = pd.read_csv(meta_csv_name, low_memory=False)

    unique_papers = raw_data_df['TestName'].unique()

    data_dfs, meta_dfs = [], []
    for i in range(len(unique_papers[:])):
        paper = unique_papers[i]

        selected_meta_df = select_rows_by_col_df(raw_meta_df, 'TestName', paper)
        selected_meta_df.columns = selected_meta_df.iloc[0] # set first row as column header
        selected_meta_df = selected_meta_df.iloc[1:, 1:] # drop first row and first col of df
        selected_meta_df = selected_meta_df.set_index(selected_meta_df.columns[0]) # set 'Name, Max, Topics' as (row) index
        selected_meta_df = selected_meta_df.dropna(axis=1) # drop cols with nan values
        no_questions = len(selected_meta_df.columns)

        selected_data_df = select_rows_by_col_df(raw_data_df, 'TestName', paper)
        selected_data_df = selected_data_df.iloc[:, 1:] # drop first col of df
        selected_data_df = selected_data_df.set_index('AnonymousStudentID') # set AnonymousStudentID as (row) index
        selected_data_df = selected_data_df.iloc[:, :no_questions] # drop first col of df
        selected_data_df = selected_data_df.dropna(axis=0) # drop rows with nan values
        # a = selected_data_df[selected_data_df.isna().any(axis=1)] # print out rows containing nan values
        # print(a)

        new_col_headers = [f'q{j+1}.{i}' for j in range(no_questions)]
        selected_data_df.columns = new_col_headers
        selected_meta_df.columns = new_col_headers

        data_dfs.append(selected_data_df)
        meta_dfs.append(selected_meta_df)
    
    return data_dfs, meta_dfs
