def split_by_question_idx(data_df, question_idx_by_pos):
    # e.g. split q0,1,5,6 out as selected_df, q2,3,4,7... as remaining_df
    selected_df = data_df.iloc[:, question_idx_by_pos]
    remaining_df = data_df.drop(data_df.columns[question_idx_by_pos], axis=1)
    return selected_df, remaining_df


def split_at_question(data_df, split_at_q):
    # e.g. if split_at_q == 5, then q0,1,2,3,4 will be in selected_df, q5 onwards in remaining_df
    selected_df = data_df.iloc[:, :split_at_q]
    remaining_df = data_df.iloc[:, split_at_q:]
    return selected_df, remaining_df
