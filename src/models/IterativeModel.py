import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix, calc_q_acc, calc_rmse

class IterativeModel:
    def __init__(self) -> None:
        self.rng = torch.Generator()


    def run(self, train_ts, test_ts, val_ts, data_dim,
                hyperparams, init_random_state,
                plot=False, save=False, step_size=25):

        '''
        if save == path_to_folder:
            save results to 'info.pt' in path_to_folder

        if plot == True:
            plot acc and nll vs iters
            if save == path_to_folder:
                save plots to 'plot.png' in path_to_folder
        '''

        self.rng.manual_seed(init_random_state)

        params, history, last_epoch = self.train(train_ts, test_ts, val_ts, data_dim,
                                                    hyperparams, step_size)

        probabilities, predictions = self.predict(test_ts, params)
        performance = self.get_performance(test_ts, predictions, probabilities)
        correctness_ts = self.get_correctness(test_ts, predictions)

        info = {'history': history,
                'model_info':{'hyperparams': hyperparams,
                                'train_epoch': last_epoch,
                                'init_random_state': init_random_state,
                                'step_size': step_size},
                'data': {'test_ts': test_ts, 
                            'data_points':{'train_ts': torch.numel(train_ts[0]),
                                            'test_ts': torch.numel(test_ts[0]),
                                            'val_ts': torch.numel(val_ts[0])}},
                'results': {'performance': performance,
                            'params': params,
                            'correctness': correctness_ts,
                            'probit': probabilities,
                            'predictions': predictions}}

        if save:
            if not os.path.exists(save):
                os.makedirs(save)
            torch.save(info, save + 'info.pt')

        if plot:
            epoch_arr = np.arange(0, last_epoch, step_size)
            self.plot_nll(history['nll'], epoch_arr, save)
            self.plot_acc(history['acc'], epoch_arr, save)

        return info


    def plot_nll(self, nll_history, epoch_arr, save):
        fig = plt.figure()
        train_nll_history = nll_history['avg train nll']
        plt.plot(range(len(train_nll_history)), train_nll_history, label='avg train nll')
        plt.plot(epoch_arr, nll_history['avg test nll'], label='avg test nll')
        plt.plot(epoch_arr, nll_history['avg val nll'], label='avg val nll')
        plt.title('average negative log likelihood')
        plt.xlabel('epoch')
        plt.legend()
        if save:
            fig.savefig(save + 'nll.png')
        plt.show()
        plt.close()


    def plot_acc(self, acc_history, epoch_arr, save):
        fig = plt.figure()
        plt.plot(epoch_arr, acc_history['train acc']/100, label='train acc')
        plt.plot(epoch_arr, acc_history['test acc']/100, label='test acc')
        plt.plot(epoch_arr, acc_history['val acc']/100, label='val acc')
        plt.title('prediction accuracy')
        plt.xlabel('epoch')
        if save:
            fig.savefig(save + 'acc.png')
        plt.show()
        plt.close()


    def train(self, train_ts):
        ...


    def predict(self, test_ts, params_ts):
        ...


    def get_performance(self, test_ts, predictions, probabilities):
        test_data, test_questions = test_ts[0], test_ts[2]
        acc = calc_acc(test_data, predictions)
        conf_matrix = calc_conf_matrix(test_data, predictions)
        q_acc = calc_q_acc(test_data, predictions, test_questions)
        rmse = calc_rmse(test_data, probabilities)
        performance = {'acc': acc, 'conf': conf_matrix, 'q_acc': q_acc, 'rmse': rmse}
        return performance


    def get_correctness(self, data_ts, predictions):
        correctness = torch.eq(data_ts[0], predictions)
        correctness_ts = torch.clone(data_ts)
        correctness_ts[0] = correctness
        return correctness_ts


    def print_progress(self, epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc):
        dp = 3
        try:
            train_nll, train_acc = round(train_nll.item()), round(train_acc,dp)
            val_nll, val_acc = round(val_nll.item()), round(val_acc,dp)
            test_nll, test_acc = round(test_nll.item()), round(test_acc,dp)
        except:
            print('NULL value')
        finally:
            print(f"epoch: {epoch:<5} train: {train_nll:<2} {train_acc:<8} val: {val_nll:<2} {val_acc:<8} test: {test_nll:<2} {test_acc:<8}")
