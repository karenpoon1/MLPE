from src.utils.data_utils.parse_paper import parse_paper

from src.utils.df_utils.combine_df import combine_df
from src.utils.df_utils.threshold_df import threshold_df
from src.utils.df_utils.binarise_df import binarise_df
from src.utils.df_utils.reset_index_df import reset_col_index, reset_row_index

def process_3papers():
    '''parse 3 papers from csv, combine papers and preprocess inc. thres, binarise, reset index'''

    # Parse raw paper
    exam_df1, meta_df1 = parse_paper('2017_1H_b1')
    exam_df2, meta_df2 = parse_paper('2017_2H_b1')
    exam_df3, meta_df3 = parse_paper('2017_3H_b1')

    # Combine papers
    combined_exam_df = combine_df([exam_df1, exam_df2, exam_df3])
    combined_meta_df = combine_df([meta_df1, meta_df2, meta_df3])

    # Threshold and binarise data
    max_scores_df = combined_meta_df.loc['Max'].astype(float)
    processed_exam_df = threshold_df(combined_exam_df, max_scores_df)
    processed_exam_df = binarise_df(processed_exam_df, max_scores_df)

    # Reset index from 0 to last
    processed_exam_df = reset_row_index(processed_exam_df)
    processed_exam_df = reset_col_index(processed_exam_df)
    processed_meta_df = reset_col_index(combined_meta_df)
    
    return processed_exam_df, processed_meta_df
