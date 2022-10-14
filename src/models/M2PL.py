import os
import torch
import math
import numpy as np
import pandas as pd

from src.config.ModelHyperparams import ModelHyperparams
from src.config.LatentHyperparams import LatentHyperparams
from src.config.LatentParams import LatentParams

from src.models.IterativeModel import IterativeModel
from src.utils.metric_utils.calc_metric import calc_acc

class M2PL(IterativeModel):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim


    def train(self, train_ts, test_ts, val_ts, data_dim, 
              hyperparams: ModelHyperparams, init_random_state, step_size):

        rate = hyperparams.rate
        iters = hyperparams.iters
        stop_method = hyperparams.stop_method
        latent_hyperparams = hyperparams.latent_hyperparams

        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        # Randomly initialise random student, question parameters
        latents = self.init_latents(data_dim, latent_hyperparams, init_random_state)
        bs, bq = latents.bs, latents.bq

        bs.requires_grad = True
        bq.requires_grad = True

        last_epoch = iters
        prev_val = 0
        prev_val_acc = 0

        bs_history, bq_history = [], []

        for epoch in range(iters):
            params = {'bs': bs, 'bq': bq}
            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:
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


    def init_latents(self, data_dim, latent_hyperparams: LatentHyperparams, random_state) -> LatentParams:
        self.rng.manual_seed(random_state)
        
        S, Q = data_dim[0], data_dim[1] # student param dimension; question param dimension
        bs0 = torch.normal(mean=latent_hyperparams.bs_mean, std=latent_hyperparams.bs_std, size=(S, 1), generator=self.rng) # std 1 for bs bq; std 0.0001 for xs xq
        bq0 = torch.normal(mean=latent_hyperparams.bq_mean, std=latent_hyperparams.bq_std, size=(Q, 1), generator=self.rng)
        latents = LatentParams(bs0, bq0)

        if self.dim > 0:
            self.rng.manual_seed(random_state+1)
            xs = torch.normal(mean=latent_hyperparams.xs_mean, std=latent_hyperparams.xs_std, size=(S, self.dim), generator=self.rng)
            xq = torch.normal(mean=latent_hyperparams.xq_mean, std=latent_hyperparams.xq_std, size=(Q, self.dim), generator=self.rng)
            latents = LatentParams(bs0, bq0, xs, xq)
        return latents


    def synthesise_from_latents(self, latents: LatentParams, synth_seed) -> pd.DataFrame:
        S, Q = latents.bs0.shape[0], latents.bq0.shape[0]

        bs0_matrix = torch.matmul(latents.bs0, torch.ones(1,Q))
        bq0_matrix = torch.matmul(latents.bq0, torch.ones(1,S)).T
        sigmoid_arg = bs0_matrix + bq0_matrix

        if self.dim > 0:
            int_matrix = torch.matmul(latents.xs, latents.xq.T)
            sigmoid_arg = bs0_matrix + bq0_matrix + int_matrix

        probit_correct = torch.sigmoid(sigmoid_arg)
        self.rng.manual_seed(synth_seed+2)
        data_ts = torch.bernoulli(probit_correct, generator=self.rng)
        
        data_df = pd.DataFrame(data_ts.detach().numpy()).astype(float)
        return data_df


    def synthesise_from_hyperparams(self, data_dim, latent_hyperparams, init_seed, synth_seed, save_dir=False):
        latents = self.init_latents(data_dim, latent_hyperparams, init_seed)
        data_df = self.synthesise_from_latents(latents, synth_seed)
        
        data_info = {'data_df': data_df, 'latents': latents.get_simplified_dict()}
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(data_info, save_dir + 'data.pt')
    
        return data_df, latents
