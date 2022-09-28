import torch
import math
import numpy as np
import pandas as pd
from src.config.ModelParams import ModelParams
from src.config.LatentParams import LatentParams

from src.models.IterativeModel import IterativeModel
from src.utils.metric_utils.calc_metric import calc_acc

class M2PL(IterativeModel):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def train(self, train_ts, test_ts, val_ts, data_dim, 
                hyperparams: ModelParams, step_size):

        rate = hyperparams.rate
        iters = hyperparams.iters
        stop_method = hyperparams.stop_method
        latent_params = hyperparams.latent_params

        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        # Randomly initialise random student, question parameters
        S, Q = data_dim[0], data_dim[1] # student param dimension; question param dimension
        bs = torch.normal(mean=latent_params.bs_mean, std=latent_params.bs_std, size=(S, self.dim+1), requires_grad=True, generator=self.rng) # std 1 for bs bq; std 0.0001 for xs xq
        bq = torch.normal(mean=latent_params.bq_mean, std=latent_params.bq_std, size=(Q, self.dim+1), requires_grad=True, generator=self.rng)

        last_epoch = iters
        prev_val = 0
        prev_val_acc = 0

        bs_history, bq_history = [], []

        for epoch in range(iters):
            params = {'bs': bs, 'bq': bq}
            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:
                # print(epoch, bs[:10])
                # print(epoch, bq[:10])

                bs_history.append(torch.clone(bs[:10]))
                bq_history.append(torch.clone(bq[:10]))
            
                test_nll = self.calc_nll(test_ts, params)
                val_nll = self.calc_nll(val_ts, params)

                if stop_method == 'nll':
                    if epoch != 0 and val_nll > prev_val:
                        last_epoch = epoch
                        break
                
                test_nll_arr[epoch//step_size], val_nll_arr[epoch//step_size] = test_nll, val_nll


                train_acc = calc_acc(train_ts[0], self.predict(train_ts, params)[1])
                test_acc = calc_acc(test_ts[0], self.predict(test_ts, params)[1])
                val_acc = calc_acc(val_ts[0], self.predict(val_ts, params)[1])

                if stop_method == 'acc':
                    if epoch != 0 and val_acc < prev_val_acc:
                        last_epoch = epoch
                        break

                train_acc_arr[epoch//step_size], test_acc_arr[epoch//step_size], val_acc_arr[epoch//step_size] = train_acc, test_acc, val_acc

                self.print_progress(epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc)

            # Gradient descent
            with torch.no_grad():
                bs -= rate * bs.grad
                bq -= rate * bq.grad

            # Zero gradients after updating
            bs.grad.zero_()
            bq.grad.zero_()

            train_nll_arr[epoch] = train_nll
            prev_val = val_nll
            prev_val_acc = val_acc

        history = {'nll': {
                        'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1],
                        'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                        'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1]
                    },
                    'acc': {
                        'train acc': np.trim_zeros(train_acc_arr, 'b'),
                        'val acc': np.trim_zeros(val_acc_arr, 'b'),
                        'test acc': np.trim_zeros(test_acc_arr, 'b'),
                    },
                    'param':{
                        'bs': bs_history,
                        'bq': bq_history
                    }
        }
        
        params = {'bs': bs, 'bq': bq}
        return params, history, last_epoch


    def calc_probit(self, data_ts, params):
        bs_test = torch.clone(params['bs'])
        bq_test = torch.clone(params['bq'])

        bs_data = torch.index_select(bs_test, 0, data_ts[1])
        bq_data = torch.index_select(bq_test, 0, data_ts[2])
        
        bs0_data = bs_data[:, 0]
        bq0_data = bq_data[:, 0]

        if self.dim == 0:
            probit_correct = torch.sigmoid(bs0_data + bq0_data)
        else:
            xs_data = bs_data[:, 1:]
            xq_data = bq_data[:, 1:]
            interactive_term = torch.sum(xs_data * xq_data, 1) # dot product between xs and xq
            probit_correct = torch.sigmoid(bs0_data + bq0_data + interactive_term)

        return probit_correct


    def calc_nll(self, data_ts, params, l=0):
        probit_correct = self.calc_probit(data_ts, params)
        nll = -torch.sum(torch.log(probit_correct**data_ts[0]) + torch.log((1-probit_correct)**(1-data_ts[0])))
        
        if self.dim > 0:
            # Regularise bq
            # xs = params['bs'][:, 1:]
            xq = params['bq'][:, 1:]
            xq_norm = torch.norm(xq, dim=1)
            penalty = torch.sum(torch.square(xq_norm))
            nll += (l * penalty)
        
        return nll


    def predict(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions


    def synthesise_data(self, data_dim, latent_params: LatentParams, random_state):
            rng = torch.Generator()
            model_dim = self.dim
            S, Q = data_dim[0], data_dim[1]

            rng.manual_seed(random_state)
            bs = torch.normal(mean=latent_params.bs_mean, std=latent_params.bs_std, size=(S, 1), requires_grad=True, generator=rng)
            bq = torch.normal(mean=latent_params.bq_mean, std=latent_params.bq_std, size=(Q, 1), requires_grad=True, generator=rng)

            bs_matrix = torch.matmul(bs, torch.ones(1,Q))
            bq_matrix = torch.matmul(bq, torch.ones(1,S)).T
            sigmoid_arg = bs_matrix + bq_matrix

            xs, xq = float('nan'), float('nan')
            if model_dim > 0:
                rng.manual_seed(random_state+1)
                xs = torch.normal(mean=latent_params.xs_mean, std=latent_params.xs_std, size=(S, model_dim), requires_grad=True, generator=rng)
                xq = torch.normal(mean=latent_params.xq_mean, std=latent_params.xq_std, size=(Q, model_dim), requires_grad=True, generator=rng)
                int_matrix = torch.matmul(xs, xq.T)
                sigmoid_arg = bs_matrix + bq_matrix + int_matrix
                
                bs = torch.concat([bs, xs], dim=1)
                bq = torch.concat([bq, xq], dim=1)

            probit_correct = torch.sigmoid(sigmoid_arg)
            rng.manual_seed(random_state+2)
            data_ts = torch.bernoulli(probit_correct, generator=rng)
            
            data_df = pd.DataFrame(data_ts.detach().numpy()).astype(float)
            true_latents = {'bs': bs, 'bq': bq}

            return data_df, true_latents
