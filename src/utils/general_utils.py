
# print df rows containing nan values
def print_nan_rows(data_df):
    data_with_nan = data_df[data_df.isna().any(axis=1)]
    print(data_with_nan)
