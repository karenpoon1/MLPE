import unittest
import pandas as pd

from src.utils.df_utils.select_rows_by_col_df import select_rows_by_col_df

class TestBinariseDf(unittest.TestCase):
    def test_select_rows_by_col_df_1(self):
        nan = float('nan')
        
        # Testcase
        data = [
            ['paper 1', 2.0, 1.0, 3.0],
            ['paper 1', 1.0, 1.0, 1.0],
            ['paper 2', 0.0, 3.0, nan],
            ['paper 2', 2.0, 1.0, nan],
            ['paper 3', 2.0, nan, nan]
        ]
        data_df = pd.DataFrame(data, columns=['PaperName', 'q1', 'q2', 'q3'])
        
        selected_df = select_rows_by_col_df(data_df, 'PaperName', 'paper 2')
        print(selected_df)

        # True
        true_data = [
            ['paper 2', 0.0, 3.0, nan],
            ['paper 2', 2.0, 1.0, nan]
        ]
        true_df = pd.DataFrame(true_data, columns=['PaperName', 'q1', 'q2', 'q3']) # set col name
        true_df.set_index(pd.Series([2,3]), inplace=True) # set row index

        print(true_df)

        self.assertTrue(selected_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()