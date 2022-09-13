# Parse csv #0 and #1 only
import pandas as pd

from src.utils.df_utils.select_rows_by_col_df import select_rows_by_col_df


def parse_paper(paper_name):
    if paper_name == 'b0':
        data_dfs, meta_dfs = parse_paper_b0()
    elif paper_name == 'b1':
        data_dfs, meta_dfs = parse_paper_b1()
    elif paper_name == 'b2':
        data_dfs, meta_dfs = parse_paper_b2()
    return data_dfs, meta_dfs

def parse_paper_b0():
    csv_name = "data/9to1_2017_GCSE_1H.csv"
    paper1_columns = ['Name'] + [f'q{i}' for i in range(1, 25)]
    paper_columns_list = [paper1_columns]
    data_row_start = 23
    no_meta_rows = 5
    return parse_paper_b01(csv_name, paper_columns_list, data_row_start, no_meta_rows)


def parse_paper_b1():
    csv_name = "data/9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv"
    paper1_columns = ['Name'] + [f'q{i}' for i in range(1, 25)]
    paper2_columns = ['Name.1'] + [f'q{i}.1' for i in range(1, 24)]
    paper3_columns = ['Name.2'] + [f'q{i}.2' for i in range(1, 24)]
    paper_columns_list = [paper1_columns, paper2_columns, paper3_columns]
    data_row_start = 6
    no_meta_rows = 5
    return parse_paper_b01(csv_name, paper_columns_list, data_row_start, no_meta_rows)


def parse_paper_b01(csv_name, paper_columns_list, data_row_start, no_meta_rows):
    raw_data = pd.read_csv(csv_name, low_memory=False)
    
    data_dfs, meta_dfs = [], []
    for paper_columns in paper_columns_list:
        data = raw_data[paper_columns]
        meta_df = data.head(no_meta_rows)
        meta_df = meta_df.set_index(paper_columns[0]) # set 'Name, Max, Topics' as (row) index

        data_df = data[data_row_start:]
        data_df = data_df[meta_df.columns] # meta_df.columns: ['q1', 'q2', ..., 'q24']
        data_df = data_df.dropna(axis=0) # drop rows with nan values, (row) index retained
        data_df = data_df.astype(float)
        
        data_dfs.append(data_df)
        meta_dfs.append(meta_df)
    
    return data_dfs, meta_dfs


def parse_paper_b2():
    data_csv_name = "data/Large Pinpoint Learning Data Set for Research.csv"
    meta_csv_name = "data/Large Pinpoint Learning Data Set for Research meta.csv"

    raw_data_df = pd.read_csv(data_csv_name, low_memory=False)
    raw_meta_df = pd.read_csv(meta_csv_name, low_memory=False)

    unique_papers = raw_data_df['TestName'].unique()

    data_dfs, meta_dfs = [], []
    for i in range(len(unique_papers[:])):
        paper = unique_papers[i]

        selected_meta_df = select_rows_by_col_df(raw_meta_df, 'TestName', paper) # select data under particular test
        selected_meta_df.columns = selected_meta_df.iloc[0] # set first row as column header
        selected_meta_df = selected_meta_df.iloc[1:, 1:] # drop first row and first col of df (exc. headers)
        selected_meta_df = selected_meta_df.set_index(selected_meta_df.columns[0]) # set 'Name, Max, Topics' as (row) index
        selected_meta_df = selected_meta_df.dropna(axis=1) # drop cols with nan values
        no_questions = len(selected_meta_df.columns)

        selected_data_df = select_rows_by_col_df(raw_data_df, 'TestName', paper)
        selected_data_df = selected_data_df.iloc[:, 1:] # drop first col of df
        selected_data_df = selected_data_df.set_index('AnonymousStudentID') # set AnonymousStudentID as (row) index
        selected_data_df = selected_data_df.iloc[:, :no_questions] # drop first col of df
        selected_data_df = selected_data_df.dropna(axis=0) # drop rows with nan values        

        new_col_headers = [f'q{j+1}.{i}' for j in range(no_questions)]
        selected_data_df.columns = new_col_headers
        selected_meta_df.columns = new_col_headers

        data_dfs.append(selected_data_df)
        meta_dfs.append(selected_meta_df)
    
    return data_dfs, meta_dfs
