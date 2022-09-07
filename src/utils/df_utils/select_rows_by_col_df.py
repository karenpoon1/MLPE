def select_rows_by_col_df(data_df, col_name, col_value):
    selected_df = data_df.loc[data_df[col_name] == col_value]
    return selected_df