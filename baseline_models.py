import torch
import numpy as np
import random
from tqdm import tqdm
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from sklearn.metrics import precision_recall_fscore_support
from utils import fit_gp_model, batch_mean_std
        
class BaseModel:
    def __init__(
            self,
            superlevelset=True,
            data_container=None,
            Xtrain=None,
            ytrain=None,
            kernel="matern"
        ):
        self.h = data_container.h
        self.superlevelset = superlevelset
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.data_container = data_container
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.bounds = torch.tensor([[0]*data_container.d, [1]*data_container.d], 
                                   device=self.device, dtype=torch.double)
        self.dtype = torch.double
        self.kernel = kernel

    def fit_model(self):
        self.GP_model = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)

    def optimize_acqf_and_get_observation(self, acq_func):
        pass

    def init_acq(self):
        pass

    def _batch_mean_std(self, X):
        y_mean, y_std = batch_mean_std(X, self.GP_model)
        return y_mean, y_std

    def predict_level_set(self, X):
        pass

    def distance_last_point(self):
        if len(self.Xtrain) > 10:
            last_point = self.Xtrain[-1].unsqueeze(0)
            dist = torch.cdist(last_point, self.Xtrain[:-1])
            return dist.min().detach().cpu().item()
        else:
            return 0
    
    def eval(self):
        lse_preds = self.predict_level_set(self.data_container.Xtest).cpu().detach().numpy()
        if self.superlevelset:
            lse_y = self.data_container.ytest > self.h
        else:
            lse_y = self.data_container.ytest < self.h
        prec, rec, f1, _ = precision_recall_fscore_support(lse_y, lse_preds, zero_division=0, labels=[False, True])
        return [f1[0], prec[0], rec[0], f1[1], prec[1], rec[1]]

    def run(self):
        pass
            
class BaseContinuousModel(BaseModel):
    def __init__(
            self, 
            superlevelset=True,
            data_container=None,
            Xtrain=None,
            ytrain=None,
            acq_name="straddle",
            epsilon=0,
            beta=1.96,
            kernel="matern"
        ):
        super().__init__(superlevelset, data_container, Xtrain, ytrain, kernel)
        self.acq_name = acq_name
        self.epsilon = epsilon
        self.beta = beta
        self.Xtest = data_container.Xtest.to(self.device, self.dtype)
        self.ytest = data_container.ytest.to(self.device, self.dtype)
        self.sampled_indices = []
        self.super = []
        self.batch_size = data_container.batch_size
        
    def predict_level_set(self, Xtest):
        y_mean, y_std = self._batch_mean_std(Xtest)
        if self.superlevelset:
            # lse_preds = y_mean > self.h - self.epsilon
            lse_preds = y_mean - self.beta*y_std > self.h + self.epsilon
        else:
            lse_preds = y_mean < self.h + self.epsilon
        return lse_preds
    
    def eval(self):
        lse_preds = torch.zeros(self.Xtest.shape[0])
        lse_preds[self.super] = 1
        all_lse_preds = self.predict_level_set(self.Xtest).cpu().detach().squeeze(1)
        lse_preds = torch.logical_or(lse_preds, all_lse_preds).bool()
        self.super = torch.nonzero(lse_preds, as_tuple=False).squeeze().tolist()
        if self.superlevelset:
            lse_y = self.ytest.cpu().detach() > self.h
        else:
            lse_y = self.ytest.cpu().detach() < self.h
        prec, rec, f1, _ = precision_recall_fscore_support(lse_y, lse_preds, zero_division=0, labels=[False, True])
        distance = self.distance_last_point()
        return [f1[0], prec[0], rec[0], f1[1], prec[1], rec[1], distance]

    def init_acq(self):
        if self.acq_name=="c2lse":
            acq_func = PhiAcqfunc(self.GP_model, self.h, self.epsilon)
        elif self.acq_name=="straddle":
            acq_func = StraddleAcqfunc(self.GP_model, self.h)
        elif self.acq_name=="TS":
            acq_func = TSAcqfunc(self.GP_model, self.h)
        return acq_func

    def optimize_acqf_and_get_observation(self, acq_func):
        idx = [i for i in range(self.Xtest.shape[0]) if i not in self.sampled_indices]
        if len(idx)>10000:
            idx = random.sample(idx, 10000)
            if self.acq_name=="TS":
                idx = random.sample(idx, 500)
        acq_val = acq_func(self.Xtest[idx].unsqueeze(1))
        del acq_func
        if len(acq_val.shape)>1:
            acq_val = acq_val.reshape((-1,))
        best_acq_val_indices = torch.topk(acq_val, k=self.batch_size).indices
        best_indices = [idx[i] for i in best_acq_val_indices]
        candidates = self.Xtest[best_indices]
        self.sampled_indices.append(best_indices)
        new_X = candidates.to(self.device)
        new_y = self.data_container.obj_func(new_X)
        self.Xtrain = torch.cat((self.Xtrain, new_X)).type(self.dtype)
        self.ytrain = torch.cat((self.ytrain, new_y.to(self.device))).type(self.dtype)
    
    def run(self, budget):
        n_iter = int(budget//self.batch_size)
        self.Xtrain = self.Xtrain.to(self.device)
        self.ytrain = self.ytrain.to(self.device)
        metrics = []
        self.fit_model()
        metrics.append(self.eval())
        for i in tqdm(range(n_iter)):
            # fit the model
            # define acquisition
            acq_func = self.init_acq()
            # optimize and get new observation
            self.optimize_acqf_and_get_observation(acq_func)
            self.fit_model()
            # evaluate
            iter_metrics = self.eval()
            metrics.append(iter_metrics)
            print(iter_metrics)
        return np.stack(metrics)
    
class PhiAcqfunc(AnalyticAcquisitionFunction):        
    def __init__(
        self,
        model,
        h,
        epsilon
    ):
        super().__init__(model=model, posterior_transform=None)
        self.h = h
        self.epsilon = epsilon

    def forward(self, X):
        mean, sigma = batch_mean_std(X.squeeze(1), self.model)
        return (sigma/torch.maximum(torch.tensor([self.epsilon]).to(X), 
                                    torch.abs(mean - self.h))).unsqueeze(-1)
               
class StraddleAcqfunc(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        h
    ):
        super().__init__(model=model, posterior_transform=None)
        self.h = h

    def forward(self, X):
        mean, sigma = batch_mean_std(X.squeeze(1), self.model)
        return (1.96*sigma - torch.abs(mean - self.h)).unsqueeze(-1)
    
class TSAcqfunc(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        h
    ):
        super().__init__(model=model, posterior_transform=None)
        self.h = h

    def forward(self, X):
        posteriors = self.model.posterior(X.squeeze(1))
        y_samples = posteriors.rsample().squeeze(0).squeeze(-1)
        del posteriors
        return -torch.abs(y_samples - self.h)

class RandomeSampling(BaseContinuousModel):
    def __init__(
            self, 
            superlevelset=True, 
            data_container=None, 
            Xtrain=None, 
            ytrain=None, 
            acq_name="straddle", 
            epsilon=0, 
            beta=1.96, 
            kernel="matern"
        ):
        super().__init__(superlevelset, data_container, Xtrain, ytrain, acq_name, epsilon, beta, kernel)

    def optimize_acqf_and_get_observation(self):
        idx = [i for i in range(self.Xtest.shape[0]) if i not in self.sampled_indices]
        best_indices = random.choices(idx, k=self.batch_size)
        candidates = self.Xtest[best_indices]
        self.sampled_indices.append(best_indices)
        new_X = candidates.to(self.device)
        new_y = self.data_container.obj_func(new_X)
        self.Xtrain = torch.cat((self.Xtrain, new_X)).type(self.dtype)
        self.ytrain = torch.cat((self.ytrain, new_y.to(self.device))).type(self.dtype)
    
    def run(self, budget):
        n_iter = int(budget//self.batch_size)
        self.Xtrain = self.Xtrain.to(self.device)
        self.ytrain = self.ytrain.to(self.device)
        metrics = []
        self.fit_model()
        metrics.append(self.eval())
        for i in tqdm(range(n_iter)):
            # optimize and get new observation
            self.optimize_acqf_and_get_observation()
            self.fit_model()
            # evaluate
            iter_metrics = self.eval()
            metrics.append(iter_metrics)
            print(iter_metrics)
        return np.stack(metrics)    