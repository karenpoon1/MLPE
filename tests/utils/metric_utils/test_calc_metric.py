import unittest
import pandas as pd
import torch

from src.utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix, calc_prec_recall, calc_rmse, calc_q_acc, calc_s_acc

class TestCalcMetric(unittest.TestCase):
    def test_calc_acc(self):
        # Testcase
        data_ts = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        predictions = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
        acc = calc_acc(data_ts, predictions)

        self.assertEqual(acc, 80.0)

    
    def test_calc_conf_matrix(self):
        #           Predicted
        #           0       1 
        # True  0|  T0      F1
        #       1|  F0      T1

        # Testcase
        data_ts = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        predictions = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
        conf_matrix = calc_conf_matrix(data_ts, predictions)

        # True
        T0, F1, F0, T1 = 1, 1, 0, 3
        true_conf_matrix = [[(T0/5)*100, (F1/5)*100], [(F0/5)*100, (T1/5)*100]]
        
        self.assertEqual(conf_matrix, true_conf_matrix)


    def test_calc_prec_recall(self):
        # precision = TP/(TP + FP)
        # recall = TP/(TP + FN)
        
        # Testcase
        conf_matrix = [[20, 30], [60, 90]]
        prec, recall = calc_prec_recall(conf_matrix)

        # True
        true_prec = (90 / (90 + 30)) * 100
        true_recall = (90 / (90 + 60)) * 100

        self.assertEqual(prec, true_prec)
        self.assertEqual(recall, true_recall)


    def test_calc_rmse(self):
        # Testcase
        data_ts = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        predictions_ts = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        rmse = calc_rmse(data_ts, predictions_ts)

        # True
        true_rmse = torch.sqrt(torch.tensor(2)/5)

        self.assertEqual(rmse, true_rmse)


#TODO: test for calc_s_acc and calc_q_acc functions


if __name__ == '__main__':
    unittest.main()