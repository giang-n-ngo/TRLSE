import itertools
import torch
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from copy import deepcopy
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, RQKernel
from gpytorch.priors.torch_priors import GammaPrior
from torch.distributions.normal import Normal
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from utils import fit_gp_model, batch_mean_std

BETA = 1.96

def init_kernel_local(dim, kernel="matern"):
    lengthscale_constraint = Interval(0.05, 4.0)
    lengthscale_prior = GammaPrior(3.0, 6.0)
    if kernel=="matern":
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, 
                ard_num_dims=dim, 
                lengthscale_constraint=lengthscale_constraint,
                lengthscale_prior=lengthscale_prior,
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    elif kernel=="rbf":
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=dim, 
                lengthscale_constraint=lengthscale_constraint,
                lengthscale_prior=lengthscale_prior,                
            )
        )
    elif kernel=="rq":
        covar_module = ScaleKernel(
            RQKernel(
                ard_num_dims=dim, 
                lengthscale_constraint=lengthscale_constraint,
                lengthscale_prior=lengthscale_prior,
            )
        )
    return covar_module
    
class C2LSEAcqfunc:        
    def __init__(
        self,
        model,
        h,
        epsilon
    ):
        self.model = model
        self.h = h
        self.epsilon = epsilon

    def __call__(self, X):
        mean, sigma = batch_mean_std(X, self.model)
        return sigma/torch.maximum(torch.tensor([self.epsilon]).to(X), 
                                   torch.abs(mean - self.h))
    
class StraddleAcqfunc:
    def __init__(
        self,
        model,
        h,
        beta=None
    ):
        self.model = model
        self.h = h
        if beta is None:
            self.beta = BETA
        else:
            self.beta = beta

    def __call__(self, X):
        mean, sigma = batch_mean_std(X, self.model)
        return self.beta*sigma - torch.abs(mean - self.h)

class TSAcqfunc:
    def __init__(
        self,
        model,
        h
    ):
        self.model = model
        self.h = h

    def __call__(self, X):
        posteriors = self.model(X)
        y_samples = posteriors.rsample()
        return -torch.abs(y_samples - self.h)

class TrustRegion:
    def __init__(
        self,
        h=0,
        epsilon=1e-2,
        Xtrain=None,
        ytrain=None,
        v_init=1e-10,
        d=10,
        centroid=None,
        n_uniform=100,
        device=torch.device('cuda'),
        v_max=1e-2,
        lengthscales_prior=None,
        superlevelset=True,
        kernel="matern",
        exploit=False
    ):
        self.device = device
        self.dtype = torch.double
        self.h = h
        self.epsilon = epsilon
        if Xtrain is not None:
            self.Xtrain = Xtrain.to(self.device, dtype=self.dtype)
            self.ytrain = ytrain.to(self.device, dtype=self.dtype)
            assert len(self.Xtrain.shape)==2, "Xtrain has incorrect number of dimensions"
            assert len(self.ytrain.shape)==2, "ytrain has incorrect number of dimensions"
        self.v_init = v_init
        self.volume = v_init
        self.d = d
        self.lengths = torch.tensor(
            [self.volume**(1/d) for i in range(d)], 
            dtype=self.dtype, device=self.device
        )
        if (self.lengths==float("inf")).any():
            raise ValueError("Lengthscales are infinite")
        if centroid is not None:
            self.centroid = centroid.to(self.device, self.dtype)
        else:
            self.centroid = torch.zeros((1, d), dtype=self.dtype, device=self.device)
        self.start = self.centroid - self.lengths/2
        self.end = self.centroid + self.lengths/2
        self.n_uniform = n_uniform
        self.v_max = v_max
        self.lengthscales_prior = lengthscales_prior
        self.region_bound = None
        self.delta = 0.05
        self.iter = 0
        self.superlevelset = superlevelset
        self.kernel = kernel
        self.exploit = exploit

    def fit_model(self):
        if self.Xtrain.shape[0] == 0:
            pass
        else:
            self.gp_model = SingleTaskGP(
                self.Xtrain, self.ytrain, 
                covar_module=init_kernel_local(self.d, self.kernel),
                likelihood=GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            )
        # self.gp_model.covar_module.base_kernel.lengthscale = self.lengthscales_prior.to(self.device, self.dtype)
        self.gp_model = self.gp_model.to(self.device, dtype=self.dtype)
        if self.Xtrain.shape[0]>0:
            mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            mll = mll.to(self.device)
            fit_gpytorch_mll(mll)
        self.gp_model.eval()
        self.Xtrain = self.Xtrain.detach()
        self.ytrain = self.ytrain.detach()

    def predict_level_set(self, X):
        y_mean, y_std = batch_mean_std(X, self.gp_model)
        lcb = y_mean - BETA*y_std
        ucb = y_mean + BETA*y_std
        if self.superlevelset is not None:
            if self.superlevelset:
                lse_preds = y_mean > self.h - self.epsilon
            else:
                lse_preds = y_mean < self.h + self.epsilon
        else:
            lse_preds = y_mean > self.h - self.epsilon
        return lse_preds.squeeze(-1), y_mean, y_std, lcb, ucb

    def update_centroid(self, lse_preds, y_mean, X_sample):
        if self.superlevelset is not None:
            if lse_preds.sum() > 0:
                new_centroid = X_sample[lse_preds].mean(dim=0)
            else:
                # if no points in target set, select the best
                if self.superlevelset:
                    new_centroid = X_sample[torch.argmax(y_mean, dim=0)[0], :]
                else:
                    new_centroid = X_sample[torch.argmin(y_mean, dim=0)[0], :]
        else:
            new_centroid = X_sample[torch.argmin(torch.abs(y_mean - self.h)[0]), :]
        return new_centroid

    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        alpha = 1
        if self.superlevelset is not None:
            if self.superlevelset:
                new_volume = self.volume*2/(1 + torch.exp(
                    - alpha*Normal(0, 1).cdf((region_lcb - self.h)/y_std[min_idx]) + \
                    alpha*Normal(0, 1).cdf((self.h - region_ucb)/y_std[max_idx])
                ))
            else:
                new_volume = self.volume*2/(1 + torch.exp(
                    alpha*Normal(0, 1).cdf((region_lcb - self.h)/y_std[min_idx]) - \
                    alpha*Normal(0, 1).cdf((self.h - region_ucb)/y_std[max_idx])
                ))
        else:
            region_mean = region_lcb/2 + region_ucb/2
            region_std = (region_ucb - region_lcb)/(2*BETA)
            new_volume = self.volume*2/(1 + torch.exp(
                -6 + 8*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
            ))
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
    def update_lengthscales(self):
        lengthscales = self.gp_model.covar_module.base_kernel.lengthscale.squeeze().detach()
        new_lengths = lengthscales*(self.volume**(1/self.d))/(torch.prod(lengthscales**(1/self.d)))
        if (new_lengths==float("inf")).any():
            print(lengthscales, 1/self.d, torch.prod(lengthscales)**(1/self.d))
            raise ValueError("Lengthscales are infinite - update")
        return new_lengths

    def update(self):
        X_sample = torch.rand(self.n_uniform, self.d, device=self.device, dtype=self.dtype)
        X_sample = self.lengths*X_sample + self.start
        lse_preds, y_mean, y_std, lcb, ucb = self.predict_level_set(X_sample)
        # new centroid
        new_centroid = self.update_centroid(lse_preds, y_mean, X_sample)
        # new lengths
        new_volume = self.update_volumne(lcb, ucb, y_std)
        new_lengths = self.update_lengthscales()
        self.centroid = new_centroid
        self.lengths = new_lengths
        self.volume = new_volume
        self.start = self.centroid - self.lengths/2
        self.end = self.centroid + self.lengths/2
        self.iter = self.iter + 1
    
    def init_acq(self, acq_name):
        if acq_name=="c2lse":
            self.acq = C2LSEAcqfunc(self.gp_model, self.h, self.epsilon)
        elif acq_name=="straddle":
            if self.exploit == "full":
                self.acq = StraddleAcqfunc(self.gp_model, self.h, 0)
            elif self.exploit == "strong":
                self.acq = StraddleAcqfunc(self.gp_model, self.h, BETA/4)
            elif self.exploit == "random":
                self.acq = RandomStraddleAcqfunc()
            else:
                if self.exploit:
                    self.acq = StraddleAcqfunc(self.gp_model, self.h, BETA/2)
                else:
                    self.acq = StraddleAcqfunc(self.gp_model, self.h)
        elif acq_name=="TS":
            self.acq = TSAcqfunc(self.gp_model, self.h)

    def compute_acq(self, X):
        acq_val = self.acq(X)
        return acq_val

    def give_random_points(self, N):
        X = torch.rand(N, self.d, device=self.device, dtype=self.dtype)
        X = (self.end - self.start)*X + self.start
        return X        

    def give_target_set(self, N):
        X = self.give_random_points(N)
        lse_preds, _, _, _, _ = self.predict_level_set(X)
        return X[lse_preds]

    def belong(self, X):
        X = X.reshape(-1, X.shape[-1])
        return torch.logical_and(
            torch.all(X >= self.start, dim=1), 
            torch.all(X <= self.end, dim=1)
        )

class AdaptiveTR_LSE:
    def __init__(
        self,
        h=0,
        epsilon=1e-2,
        n_regions=10,
        d=10,
        v_init=1e-10,
        batch_size=10,
        acq_name="straddle",
        n_eval = 10000,
        v_max=1e-2,
        lengthscales_prior=None,
        superlevelset=True,
        kernel="matern"
    ):
        self.h = h
        self.epsilon = epsilon
        self.n_regions = n_regions
        self.regions = dict([(i, None) for i in range(self.n_regions)])
        self.d = d
        self.v_init = v_init
        self.batch_size=batch_size
        self.v_min = v_init/2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Device: {self.device}")
        self.dtype = torch.double
        self.n_uniform = 100
        self.acq_name = acq_name
        self.n_eval = n_eval
        self.v_max = v_max
        self.Xtrain, self.ytrain = None, None
        self.X_init, self.y_init = None, None
        self.lengthscales_prior = lengthscales_prior
        self.super, self.sub = [], []
        self.superlevelset = superlevelset
        self.restart = 0
        self.sampled_indices = []
        self.kernel = kernel
        self.local_acq_val, self.global_acq_val = [], []
        self.global_local_acq_val = []

    def init_region(
            self, 
            centroid, 
            region_idx,
            superlevelset=None
        ):
        # initialize region
        if superlevelset is None:
            superlevelset = self.superlevelset
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            superlevelset,
            self.kernel
        )
        self.region_data_selection(region_idx)

    def reinit_region(self, region_idx, superlevelset=None):
        del self.regions[region_idx]
        # fit GP init
        self.global_gp_init = fit_gp_model(self.X_init, self.y_init, self.device, self.dtype, self.kernel)
        # draw realization
        pool_indices = torch.ones(self.Xtest.shape[0], dtype=bool)
        pool_indices[self.sampled_indices] = False
        # handle OOM case
        if self.Xtest.shape[0] - len(self.sampled_indices) > 10000:
            true_indices = torch.where(pool_indices)[0]
            indices_to_flip = np.random.choice(
                true_indices, 
                size=self.Xtest.shape[0] - len(self.sampled_indices) - 10000, 
                replace=False
            )
            pool_indices[indices_to_flip] = False
        centroid_candidates = self.Xtest[pool_indices]
        posteriors = self.global_gp_init(centroid_candidates)
        # y_samples = posteriors.rsample()
        # # optimize realization
        # if superlevelset is None:
        #     if self.superlevelset:
        #         centroid = centroid_candidates[torch.argmax(y_samples)]
        #     else:
        #         centroid = centroid_candidates[torch.argmin(y_samples)]
        # else:
        #     if superlevelset:
        #         centroid = centroid_candidates[torch.argmax(y_samples)]
        #     else:
        #         centroid = centroid_candidates[torch.argmin(y_samples)]
        y_std = posteriors.variance**0.5
        centroid = centroid_candidates[torch.argmax(y_std)]
        y_centroid = self.f_eval(centroid.unsqueeze(0)).to(self.ytrain)
        self.Xtrain = torch.cat((self.Xtrain, centroid.unsqueeze(0)), dim=0)
        self.ytrain = torch.cat((self.ytrain, y_centroid), dim=0)
        self.init_region(centroid, region_idx, superlevelset)
        self.X_init = torch.cat((self.X_init, centroid.unsqueeze(0)), dim=0)
        self.y_init = torch.cat((self.y_init, y_centroid), dim=0)
        self.restart += 1

    def region_data_selection(self, i):
        region = self.regions[i]
        selected_indices = torch.stack((
            (region.start - region.lengths/2<= self.Xtrain).all(1), 
            (self.Xtrain <= region.end + region.lengths/2).all(1)
        ), dim=0).all(0)
        region.Xtrain = deepcopy(self.Xtrain[selected_indices].detach())
        region.ytrain = deepcopy(self.ytrain[selected_indices].detach())
        region.fit_model()

    def check_region_overlap(self, i, j):
        start_i, end_i = self.regions[i].start, self.regions[i].end
        start_j, end_j = self.regions[j].start, self.regions[j].end
        overlaps = [max(0, min(end_i[k], end_j[k]) - max(start_i[k], start_j[k])) \
                    for k in range(self.d)]
        overlap_volume = torch.prod(torch.tensor(overlaps))
        return overlap_volume/self.regions[i].volume, overlap_volume/self.regions[j].volume

    def __get_centroids__(self):
        centroids = [self.regions[i].centroid for i in range(self.n_regions)]
        centroids = torch.stack(centroids, dim=0)
        return centroids

    def allocate_point(self, X, selection="dist", std=None):
        # std must be of shape N*n_regions
        centroids = self.__get_centroids__()
        if selection=="dist":
            dist = torch.cdist(centroids, X)
            indices = torch.min(dist, dim=0).indices
        elif selection=="std":
            indices = torch.min(std, dim=1).indices
        return indices

    def next_point_selection(self):
        X_candidates, acq_val = [], []
        for i in range(self.n_regions):
            self.regions[i].init_acq(self.acq_name)
            i_X_candidates = self.regions[i].give_random_points(
                max(int(self.n_uniform*np.log10(2*self.regions[i].volume/self.v_init)), 1)
            )
            acq_val.append(self.regions[i].compute_acq(i_X_candidates))
            X_candidates.append(i_X_candidates)
        X_candidates = torch.cat(X_candidates, dim=0)
        acq_val = torch.cat(acq_val, dim=0).squeeze(-1)
        self.local_acq_val.append(acq_val.max().item())
        selected_indices = torch.topk(acq_val, self.batch_size).indices
        selected_candidates = X_candidates[selected_indices]
        selected_local_acq_vals = acq_val[selected_indices]
        print("Local acq values max, mean, min:", selected_local_acq_vals.max().item(), selected_local_acq_vals.mean().item(), selected_local_acq_vals.min().item())
        # calculate global acq values at these points for logging
        ## check if self.global_gp_init exists, if not create it
        if not hasattr(self, "global_gp_init"):
            self.global_gp_init = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)
        global_acq = StraddleAcqfunc(self.global_gp_init, self.h)
        global_acq_val = global_acq(selected_candidates).squeeze(-1)
        print("Global-local acq values max, mean, min:", global_acq_val.max().item(), global_acq_val.mean().item(), global_acq_val.min().item())
        self.global_local_acq_val.append(global_acq_val.max().item())
        del global_acq

        func_val = self.f_eval(selected_candidates).to(self.ytrain)
        self.Xtrain = torch.cat((self.Xtrain, selected_candidates), dim=0)
        self.ytrain = torch.cat((self.ytrain, func_val), dim=0)
        return selected_candidates
    
    def region_union_bound(self, X):
        lcb, ucb, y_std = [], [], []
        for i in range(self.n_regions):
            if self.regions[i].Xtrain.shape[0]>0:
                i_y_mean, i_y_std = batch_mean_std(X, self.regions[i].gp_model)
                lcb.append(i_y_mean - BETA*i_y_std)
                ucb.append(i_y_mean + BETA*i_y_std)
                y_std.append(i_y_std)
        lcb = torch.stack(lcb, dim=1).squeeze(-1)
        ucb = torch.stack(ucb, dim=1).squeeze(-1)
        y_std = torch.stack(y_std, dim=1).squeeze(-1)
        indices = self.allocate_point(X, "std", y_std)
        # indices = self.allocate_point(X)
        lcb = lcb.gather(1, indices.unsqueeze(1)).squeeze(1)
        ucb = ucb.gather(1, indices.unsqueeze(1)).squeeze(1)
        y_std = y_std.gather(1, indices.unsqueeze(1)).squeeze(1)
        return lcb, ucb, y_std

    def predict_level_set(self, X):
        lcb, ucb, _ = self.region_union_bound(X)
        if self.superlevelset:
            lse_preds = lcb > self.h
        else:
            lse_preds = ucb < self.h
        return lse_preds, lcb, ucb

    def update_set(self):
        lcb, ucb, y_std = self.region_union_bound(self.Xtest)
        intersect = torch.logical_or(
            torch.logical_and(self.lcb <= lcb, lcb < self.ucb),
            torch.logical_and(self.lcb < ucb, ucb <= self.ucb)
        )
        self.std = (self.ucb - self.lcb)/3.92
        self.lcb[intersect] = torch.maximum(self.lcb, lcb)[intersect]
        self.ucb[intersect] = torch.minimum(self.ucb, ucb)[intersect]
        std_indices = self.std > y_std
        self.lcb[torch.logical_and(~intersect, std_indices)] = lcb[torch.logical_and(~intersect, std_indices)]
        self.lcb[torch.logical_and(~intersect, ~std_indices)] = self.lcb[torch.logical_and(~intersect, ~std_indices)]
        self.ucb[torch.logical_and(~intersect, std_indices)] = ucb[torch.logical_and(~intersect, std_indices)]
        self.ucb[torch.logical_and(~intersect, ~std_indices)] = self.ucb[torch.logical_and(~intersect, ~std_indices)]        
        new_super = (self.lcb > self.h).nonzero().squeeze(1).detach().cpu().tolist()
        new_sub = (self.ucb < self.h).nonzero().squeeze(1).detach().cpu().tolist()
        self.super = list(set(self.super).union(set(new_super)))
        self.sub = list(set(self.sub).union(set(new_sub)))
        self.unclassified = [idx for idx in range(self.Xtest.shape[0]) if \
                             idx not in self.super and idx not in self.sub]
    
    def eval(self):
        X = deepcopy(self.Xtrain.detach().cpu())
        y = deepcopy(self.ytrain.detach().cpu())
        global_gp = fit_gp_model(X, y, self.device, self.dtype, self.kernel)
        y_mean, _ = batch_mean_std(self.Xtest, global_gp)
        del global_gp, X, y
        lse_preds = (y_mean > self.h - self.epsilon).detach().cpu().numpy()
        lse_y = (self.ytest > self.h).detach().cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(
            lse_y, 
            lse_preds, 
            zero_division=0,
            labels=[False, True]
        )
        return f1, prec, rec
    
    def plot_regions(self, iteration, local_X, savefig=True):
        plt.ioff()
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        indices = self.y_grid_plot.squeeze(1)>self.h
        plt.scatter(self.X_grid_plot[indices, 0], self.X_grid_plot[indices, 1], color="orange")
        mask = ~torch.all(self.Xtrain==local_X, dim=1)
        plt.scatter(
            local_X[:, 0].cpu().detach(), 
            local_X[:, 1].cpu().detach(), 
            color="blue",
            marker="*"
        )
        plt.scatter(
            self.Xtrain[mask, 0].cpu().detach(),
            self.Xtrain[mask, 1].cpu().detach(),
            s=2,
            color="red"
        )
        currentAxis = plt.gca()
        for i in range(self.n_regions):
            centroid = self.regions[i].centroid.detach().cpu().numpy()
            width = self.regions[i].lengths[0].detach().cpu().item()
            height = self.regions[i].lengths[1].detach().cpu().item()
            currentAxis.add_patch(
                Rectangle(
                    (centroid[0] - width/2, centroid[1] - height/2), 
                    width, 
                    height, 
                    facecolor="none", 
                    ec="forestgreen", 
                    lw=0.55
                )
            )
        if savefig:
            os.makedirs(f"images/{self.data_name}", exist_ok=True)
            plt.savefig(f"images/{self.data_name}/regions_{iteration}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def merge_regions(self, i, j):
        if self.regions[i].ytrain.shape[0] > self.regions[j].ytrain.shape[0]:
            retain_idx = i
            remove_idx = j
        else:
            retain_idx = j
            remove_idx = i
        new_start = torch.minimum(self.regions[i].start, self.regions[j].start)
        new_end = torch.maximum(self.regions[i].end, self.regions[j].end)
        new_lengths = new_end - new_start
        new_centroid = (new_start + new_end)/2
        lengthscales = self.regions[remove_idx].gp_model.covar_module.base_kernel.lengthscale.squeeze()
        lengthscales_prior = self.regions[remove_idx].lengthscales_prior
        remove_iter = self.regions[remove_idx].iter
        del self.regions[remove_idx]
        self.regions[remove_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            new_centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            superlevelset=self.superlevelset,
            kernel=self.kernel
        )
        new_volume = min(torch.prod(new_lengths).cpu().item(), self.v_max)
        self.regions[remove_idx].start = new_start
        self.regions[remove_idx].end = new_end
        self.regions[remove_idx].volume = new_volume
        self.regions[remove_idx].lengths = lengthscales*(self.regions[remove_idx].volume**(1/self.d))/(torch.prod(lengthscales**(1/self.d)))
        self.regions[remove_idx].lengthscales_prior = lengthscales_prior
        self.region_data_selection(remove_idx)
        self.regions[remove_idx].iter = remove_iter
        self.reinit_region(retain_idx)

    def check_blank_data(self):
        blank = []
        for i in range(self.n_regions):
            if self.regions[i].Xtrain.shape[0]==0:
                blank.append(i)
        print(blank)

    def total_volume(self):
        volume = 0
        for _, region in self.regions.items():
            volume += region.volume
        return volume

    def target_count(self):
        point_count = []
        for region in self.regions.values():
            point_count.append((region.ytrain > self.h).sum().detach().item())
        return point_count
    
    def total_count(self):
        point_count = []
        for region in self.regions.values():
            point_count.append(region.ytrain.shape[0])
        return sum(point_count)

    def update_region(self, i):
        self.regions[i].update()
        self.region_data_selection(i)

    def discard_region(self, i):
        if self.regions[i].volume < self.v_min or self.regions[i].Xtrain.shape[0]==0:
            self.reinit_region(i, self.regions[i].superlevelset)

    def eval_print_result(self, i):
        if i==0:
            col_names = ["i", "Sub", "Super", "Volume", "%", "#N", "Sub", "Super", "Time"]
            print("{:<3}| {:<16}| {:<16}| {:<8}| {:<5}| {:<4}| {:<5}| {:<5}| {:<2}".format(*col_names))
            print("   | {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}|".format(*["F1", "Prec", "Rec", "F1", "Prec", "Rec"]))
        f1, prec, rec = self.eval()
        total_volume = self.total_volume()
        total_target_count = sum(self.target_count())
        total_point_count = self.total_count()
        row_data = [
            i, 
            round(f1[0], 2), 
            round(prec[0], 2),
            round(rec[0], 2),
            round(f1[1], 2), 
            round(prec[1], 2),
            round(rec[1], 2),
            '{:.2E}'.format(total_volume),
            round(total_target_count*100/total_point_count, 1),
            self.ytrain.shape[0],
            len(self.sub),
            len(self.super),
            round(time.time() - self.time)
        ]
        formatted_row = "{:<3}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<8}| {:<5}| {:<4}| {:<5}| {:<5}| {:<2}".format(*row_data)
        print(formatted_row)
        return [f1, prec, rec]

    def setup_model(self, X_init, y_init, init_indices, f_eval, dataloader):
        self.f_eval = f_eval
        self.data_name = dataloader.data_name
        self.Xtest = dataloader.Xtest.to(self.device, self.dtype)
        self.ytest = dataloader.ytest.to(self.device, self.dtype)
        self.sampled_indices.extend(init_indices)
        self.unclassified = [i for i in range(self.Xtest.shape[0]) if i not in init_indices]
        for i in init_indices:
            if dataloader.ytest[i] > self.h:
                self.super.append(i)
            else:
                self.sub.append(i)
        self.lcb = torch.tensor([float('-inf')]*self.Xtest.shape[0]).to(self.device)
        self.ucb = torch.tensor([float('inf')]*self.Xtest.shape[0]).to(self.device)
        self.Xtrain = X_init.to(self.device, self.dtype)
        self.X_init = X_init.to(self.device, self.dtype)
        self.ytrain = y_init.to(self.device, self.dtype)
        self.y_init = y_init.to(self.device, self.dtype)
        # initialize regions
        assert (X_init.shape[0]==self.n_regions) and (y_init.shape[0]==self.n_regions), "Init wrong"
        if self.kernel!="linear":
            temp_gp = fit_gp_model(self.X_init, self.y_init, self.device, self.dtype, self.kernel)
            self.lengthscales_prior = temp_gp.covar_module.base_kernel.lengthscale.detach().cpu()
            del temp_gp
        for i in range(self.n_regions):
            self.init_region(self.Xtrain[i, :], i, True)
        if self.d==2:
            self.X_grid_plot, self.y_grid_plot = dataloader.gen_grid_XY(500)
        self.budget = int(dataloader.budget/dataloader.batch_size)

    def run(self, budget):
        # loop
        all_metrics = []
        for i in range(budget):
            current_time = time.time()
            ## next point eval
            local_X = self.next_point_selection()
            if self.d==2:
                self.plot_regions(i, local_X)
            ## update region
            for j in range(self.n_regions):
                self.update_region(j)
                self.discard_region(j)
            for j, k in list(itertools.product(range(self.n_regions), range(self.n_regions))):
                if j==k:
                    continue
                overlap_j, overlap_k = self.check_region_overlap(j, k)
                if overlap_j > 0.8 or overlap_k > 0.8: # merge if overlap too much
                    self.merge_regions(j, k)
            self.update_set()
            temp_gp = fit_gp_model(self.X_init, self.y_init, self.device, self.dtype, self.kernel)
            if self.kernel!="linear":
                self.lengthscales_prior = temp_gp.covar_module.base_kernel.lengthscale.detach().cpu()
            if i%5==0:
            ## evaluate
                iter_metrics = self.eval_print_result(i)
                all_metrics.append(iter_metrics)
                print("Run time: ", round(time.time() - current_time))
                current_time = time.time()
        if i%5!=0:
        ## evaluate
            iter_metrics = self.eval_print_result(i)
            all_metrics.append(iter_metrics)
        return np.array(all_metrics)
    
class BoundaryLSE(AdaptiveTR_LSE):
    def __init__(
        self, 
        h=0, 
        epsilon=0.01, 
        n_regions=10, 
        d=10, 
        v_init=1e-10, 
        batch_size=10, 
        acq_name="straddle", 
        n_eval=10000, 
        v_max=0.01, 
        lengthscales_prior=None, 
        kernel="matern"
    ):
        super().__init__(
            h, epsilon, n_regions, d, v_init, 
            batch_size, acq_name, n_eval, v_max, 
            lengthscales_prior, None, kernel
        )
        self.iter_global_acq_val = []

    def init_acq(self):
        if self.acq_name=="c2lse":
            self.acq = C2LSEAcqfunc(self.global_gp_init, self.h, self.epsilon)
        elif self.acq_name=="straddle":
            self.acq = StraddleAcqfunc(self.global_gp_init, self.h)
        elif self.acq_name=="TS":
            self.acq = TSAcqfunc(self.global_gp_init, self.h)

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            None,
            kernel=self.kernel
        )
        self.region_data_selection(region_idx)

    def find_pool_outside(self, max_candidates=10000):
        pool_indices = torch.ones(self.Xtest.shape[0], dtype=bool)
        pool_indices[self.sampled_indices] = False
        belong_regions = []
        for _, region in self.regions.items():
            belong_region = region.belong(self.Xtest)
            belong_regions.append(belong_region)
        belong_regions = torch.stack(belong_regions)
        belong_regions = belong_regions.any(dim=0)
        
        assert belong_regions.shape[0]==self.Xtest.shape[0], "Wrong belong dim"
        pool_indices[belong_regions] = False
        # handle OOM case 
        if self.Xtest.shape[0] - len(self.sampled_indices) > max_candidates:
            true_indices = torch.where(pool_indices)[0]
            indices_to_flip = np.random.choice(
                true_indices, 
                size=self.Xtest.shape[0] - len(self.sampled_indices) - max_candidates, 
                replace=False
            )
            pool_indices[indices_to_flip] = False
        return pool_indices
    
    def distance_last_point(self):
        if len(self.Xtrain) > 10:
            last_point = self.Xtrain[-1].unsqueeze(0)
            dist = torch.cdist(last_point, self.Xtrain[:-1])
            return dist.min().detach().cpu().item()
        else:
            return 0

    def eval(self):
        X = deepcopy(self.Xtrain.detach().cpu())
        y = deepcopy(self.ytrain.detach().cpu())
        global_gp = fit_gp_model(X, y, self.device, self.dtype, self.kernel)
        y_mean, y_std = batch_mean_std(self.Xtest, global_gp)
        del global_gp, X, y
        lse_preds = (y_mean > self.h - self.epsilon).detach().cpu().numpy()
        lse_y = (self.ytest > self.h).detach().cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(
            lse_y, 
            lse_preds, 
            zero_division=0,
            labels=[False, True]
        )
        distance = self.distance_last_point()
        return f1, prec, rec, distance

    def reinit_region(self, region_idx):
        del self.regions[region_idx]
        # fit GP init
        self.global_gp_init = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)
        # draw realization
        pool_indices = self.find_pool_outside()
        centroid_candidates = self.Xtest[pool_indices]
        pool_indices = torch.where(pool_indices)[0]
        assert centroid_candidates.shape[0] <= 10000, "Pool size too large, will cause OOM!"
        self.init_acq()
        # print(centroid_candidates.shape)
        acq_val = self.acq(centroid_candidates)
        self.iter_global_acq_val.append(acq_val.max().item())
        candidate_idx = torch.argmax(acq_val)
        centroid = centroid_candidates[candidate_idx]
        y_centroid = self.f_eval(centroid.unsqueeze(0)).to(self.ytrain)
        self.Xtrain = torch.cat((self.Xtrain, centroid.unsqueeze(0)), dim=0)
        self.ytrain = torch.cat((self.ytrain, y_centroid), dim=0)
        self.init_region(centroid, region_idx)
        self.X_init = torch.cat((self.X_init, centroid.unsqueeze(0)), dim=0)
        self.y_init = torch.cat((self.y_init, y_centroid), dim=0)
        self.sampled_indices.append(pool_indices[candidate_idx].item())
        self.restart += 1

    def merge_regions(self, i, j, local_exploit=""):
        if self.regions[i].ytrain.shape[0] > self.regions[j].ytrain.shape[0]:
            retain_idx = i
            remove_idx = j
        else:
            retain_idx = j
            remove_idx = i
        new_start = torch.minimum(self.regions[i].start, self.regions[j].start)
        new_end = torch.maximum(self.regions[i].end, self.regions[j].end)
        new_lengths = new_end - new_start
        new_centroid = (new_start + new_end)/2
        lengthscales = self.regions[remove_idx].gp_model.covar_module.base_kernel.lengthscale.squeeze()
        lengthscales_prior = self.regions[remove_idx].lengthscales_prior
        remove_iter = self.regions[remove_idx].iter
        del self.regions[remove_idx]
        self.regions[remove_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            new_centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            None,
            None,
            self.kernel,
            local_exploit
        )
        new_volume = min(torch.prod(new_lengths).cpu().item(), self.v_max)
        self.regions[remove_idx].start = new_start
        self.regions[remove_idx].end = new_end
        self.regions[remove_idx].volume = new_volume
        self.regions[remove_idx].lengths = lengthscales*(self.regions[remove_idx].volume**(1/self.d))/(torch.prod(lengthscales**(1/self.d)))
        self.regions[remove_idx].lengthscales_prior = lengthscales_prior
        self.region_data_selection(remove_idx)
        self.regions[remove_idx].iter = remove_iter
        self.reinit_region(retain_idx)

    def discard_region(self, i):
        if self.regions[i].volume < self.v_min or self.regions[i].Xtrain.shape[0]==0:
            self.reinit_region(i)

    def setup_model(self, X_init, y_init, f_eval, dataloader):
        self.f_eval = f_eval
        self.data_name = dataloader.data_name
        self.Xtest = dataloader.Xtest.to(self.device, self.dtype)
        self.ytest = dataloader.ytest.to(self.device, self.dtype)
        self.lcb = torch.tensor([float('-inf')]*self.Xtest.shape[0]).to(self.device, self.dtype)
        self.ucb = torch.tensor([float('inf')]*self.Xtest.shape[0]).to(self.device, self.dtype)
        self.Xtrain = X_init.to(self.device, self.dtype)
        self.X_init = X_init.to(self.device, self.dtype)
        self.ytrain = y_init.to(self.device, self.dtype)
        self.y_init = y_init.to(self.device, self.dtype)
        # initialize regions
        assert (X_init.shape[0]==self.n_regions) and (y_init.shape[0]==self.n_regions), f"Init wrong, got {X_init.shape[0]} and {y_init.shape[0]} for {self.n_regions}"
        if self.kernel!="linear":
            temp_gp = fit_gp_model(self.X_init, self.y_init, self.device, self.dtype, self.kernel)
            self.lengthscales_prior = temp_gp.covar_module.base_kernel.lengthscale.detach().cpu()
            del temp_gp
        for i in range(self.n_regions):
            self.init_region(self.Xtrain[i, :], i)
        if self.d==2:
            self.X_grid_plot, self.y_grid_plot = dataloader.gen_grid_XY(500)
        self.budget = int(dataloader.budget/dataloader.batch_size)
        
    def eval_print_result(self, i):
        if i==0:
            col_names = ["i", "Sub", "Super", "Volume", "#N", "Restart", "Time"]
            print("{:<3}| {:<16}| {:<16}| {:<8}| {:<5}| {:<7}| {:<2}".format(*col_names))
            print("   | {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}|".format(*["F1", "Prec", "Rec", "F1", "Prec", "Rec"]))
        f1, prec, rec, distance = self.eval()
        total_volume = self.total_volume()
        row_data = [
            i, 
            round(f1[0], 2), 
            round(prec[0], 2),
            round(rec[0], 2),
            round(f1[1], 2), 
            round(prec[1], 2),
            round(rec[1], 2),
            '{:.2E}'.format(total_volume),
            self.ytrain.shape[0],
            self.restart,
            round(time.time() - self.time)
        ]
        formatted_row = "{:<3}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<4}| {:<8}| {:<7}| {:<2}".format(*row_data)
        print(formatted_row)
        return [
            f1[0], prec[0], rec[0], 
            f1[1], prec[1], rec[1], 
            total_volume, self.ytrain.shape[0], self.restart, distance
        ]

    def run(self, budget):
        # loop
        all_metrics = []
        budget = self.budget if budget is None else budget
        self.time = time.time()
        all_metrics.append(self.eval_print_result(0))
        for i in tqdm(range(1, budget+1)):
            self.time = time.time()
            ## next point eval
            local_X = self.next_point_selection()
            if self.d==2:
                self.plot_regions(i, local_X)
            ## update region
            for j in range(self.n_regions):
                self.update_region(j)
                self.discard_region(j)
            self.global_acq_val.append(np.min(self.iter_global_acq_val) if len(self.iter_global_acq_val)>0 else None)
            self.iter_global_acq_val = []
            ### merge and re-initialize
            for j, k in list(itertools.product(range(self.n_regions), range(self.n_regions))):
                if j==k:
                    continue
                overlap_j, overlap_k = self.check_region_overlap(j, k)
                if overlap_j > 0.5 or overlap_k > 0.5: # merge if overlap too much
                    self.merge_regions(j, k)
            temp_gp = fit_gp_model(self.X_init, self.y_init, self.device, self.dtype, self.kernel)
            self.lengthscales_prior = temp_gp.covar_module.base_kernel.lengthscale.detach().cpu()
            del temp_gp
            ## evaluate
            iter_metrics = self.eval_print_result(i)
            all_metrics.append(iter_metrics)
        return np.array(all_metrics)
    
class BoundaryLSE_RandomTR(BoundaryLSE):
    def reinit_region(self, region_idx):
        del self.regions[region_idx]
        # draw realization
        pool_indices = self.find_pool_outside(1000)
        centroid_candidates = self.Xtest[pool_indices]
        pool_indices = torch.where(pool_indices)[0]
        assert centroid_candidates.shape[0] <= 1000, "Pool size too large, will cause OOM!"
        candidate_idx = random.sample(range(centroid_candidates.shape[0]), 1)[0]
        centroid = centroid_candidates[candidate_idx]
        y_centroid = self.f_eval(centroid.unsqueeze(0)).to(self.ytrain)
        self.Xtrain = torch.cat((self.Xtrain, centroid.unsqueeze(0)), dim=0)
        self.ytrain = torch.cat((self.ytrain, y_centroid), dim=0)
        self.init_region(centroid, region_idx)
        self.X_init = torch.cat((self.X_init, centroid.unsqueeze(0)), dim=0)
        self.y_init = torch.cat((self.y_init, y_centroid), dim=0)
        self.sampled_indices.append(pool_indices[candidate_idx].item())
        self.restart += 1

class TrustRegion_GlobalGP(TrustRegion):
    def __init__(
        self,
        gp_model,
        h=0,
        epsilon=1e-2,
        v_init=1e-10,
        d=10,
        centroid=None,
        n_uniform=100,
        device=torch.device('cuda'),
        v_max=1e-2,
        superlevelset=True,
        kernel="matern"
    ):
        super().__init__(
            h, epsilon, None, None, v_init, d, centroid, n_uniform, device, v_max, None, superlevelset, kernel
        )
        self.gp_model = gp_model

    def fit_model(self):
        raise NotImplementedError("This TR cannot fit its own model. Check code")

class BoundaryLSE_OneGP(BoundaryLSE):

    def init_acq(self):
        if self.acq_name=="c2lse":
            self.acq = C2LSEAcqfunc(self.global_gp, self.h, self.epsilon)
        elif self.acq_name=="straddle":
            self.acq = StraddleAcqfunc(self.global_gp, self.h)
        elif self.acq_name=="TS":
            self.acq = TSAcqfunc(self.global_gp, self.h)

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion_GlobalGP(
            self.global_gp,
            self.h,
            self.epsilon,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            None,
            kernel=self.kernel
        )

    def reinit_region(self, region_idx):
        del self.regions[region_idx]
        # draw realization
        pool_indices = self.find_pool_outside()
        centroid_candidates = self.Xtest[pool_indices]
        pool_indices = torch.where(pool_indices)[0]
        assert centroid_candidates.shape[0] <= 10000, "Pool size too large, will cause OOM!"
        self.init_acq()
        acq_val = self.acq(centroid_candidates)
        self.iter_global_acq_val.append(acq_val.max().item())
        candidate_idx = torch.argmax(acq_val)
        centroid = centroid_candidates[candidate_idx]
        y_centroid = self.f_eval(centroid.unsqueeze(0)).to(self.ytrain)
        self.Xtrain = torch.cat((self.Xtrain, centroid.unsqueeze(0)), dim=0)
        self.ytrain = torch.cat((self.ytrain, y_centroid), dim=0)
        self.init_region(centroid, region_idx)
        self.sampled_indices.append(pool_indices[candidate_idx].item())
        self.restart += 1
        self.global_gp = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)

    def merge_regions(self, i, j):
        if self.regions[i].volume > self.regions[j].volume:
            retain_idx = i
            remove_idx = j
        else:
            retain_idx = j
            remove_idx = i
        new_start = torch.minimum(self.regions[i].start, self.regions[j].start)
        new_end = torch.maximum(self.regions[i].end, self.regions[j].end)
        new_lengths = new_end - new_start
        new_centroid = (new_start + new_end)/2
        lengthscales = self.regions[remove_idx].gp_model.covar_module.base_kernel.lengthscale.squeeze()
        remove_iter = self.regions[remove_idx].iter
        del self.regions[remove_idx]
        self.regions[remove_idx] = TrustRegion_GlobalGP(
            self.global_gp,
            self.h,
            self.epsilon,
            self.v_init,
            self.d,
            new_centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            None,
            self.kernel
        )
        new_volume = min(torch.prod(new_lengths).cpu().item(), self.v_max)
        self.regions[remove_idx].start = new_start
        self.regions[remove_idx].end = new_end
        self.regions[remove_idx].volume = new_volume
        self.regions[remove_idx].lengths = lengthscales*(self.regions[remove_idx].volume**(1/self.d))/(torch.prod(lengthscales**(1/self.d)))
        self.regions[remove_idx].iter = remove_iter
        self.reinit_region(retain_idx)
    
    def setup_model(self, X_init, y_init, f_eval, dataloader):
        self.f_eval = f_eval
        self.data_name = dataloader.data_name
        self.Xtest = dataloader.Xtest.to(self.device, self.dtype)
        self.ytest = dataloader.ytest.to(self.device, self.dtype)
        self.lcb = torch.tensor([float('-inf')]*self.Xtest.shape[0]).to(self.device, self.dtype)
        self.ucb = torch.tensor([float('inf')]*self.Xtest.shape[0]).to(self.device, self.dtype)
        self.Xtrain = X_init.to(self.device, self.dtype)
        self.ytrain = y_init.to(self.device, self.dtype)
        # initialize regions
        assert (X_init.shape[0]==self.n_regions) and (y_init.shape[0]==self.n_regions), "Init wrong"
        self.global_gp = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)
        for i in range(self.n_regions):
            self.init_region(self.Xtrain[i, :], i)
        if self.d==2:
            self.X_grid_plot, self.y_grid_plot = dataloader.gen_grid_XY(500)
        self.budget = int(dataloader.budget/dataloader.batch_size)

    def eval(self):
        X = deepcopy(self.Xtrain.detach().cpu())
        y = deepcopy(self.ytrain.detach().cpu())
        y_mean, _ = batch_mean_std(self.Xtest, self.global_gp)
        del X, y
        lse_preds = (y_mean > self.h - self.epsilon).detach().cpu().numpy()
        lse_y = (self.ytest > self.h).detach().cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(
            lse_y, 
            lse_preds, 
            zero_division=0,
            labels=[False, True]
        )
        distance = self.distance_last_point()
        return f1, prec, rec, distance

    def discard_region(self, i):
        if self.regions[i].volume < self.v_min:
            self.reinit_region(i)

    def target_count(self):
        point_count = []
        for region in self.regions.values():
            point_count.append(
                1
            )
        return point_count
    
    def total_count(self):
        point_count = []
        for region in self.regions.values():
            point_count.append(1)
        return sum(point_count)

    def run(self, budget):
        # loop
        all_metrics = []
        budget = self.budget if budget is None else budget
        self.time = time.time()
        all_metrics.append(self.eval_print_result(0))
        for i in tqdm(range(1, budget+1)):
            self.time = time.time()
            ## next point eval
            local_X = self.next_point_selection()
            self.global_gp = fit_gp_model(self.Xtrain, self.ytrain, self.device, self.dtype, self.kernel)
            if self.d==2:
                self.plot_regions(i, local_X)
            ## update region
            for j in range(self.n_regions):
                self.regions[j].update()
                self.discard_region(j)
            ### merge and re-initialize
            for j, k in list(itertools.product(range(self.n_regions), range(self.n_regions))):
                if j==k:
                    continue
                overlap_j, overlap_k = self.check_region_overlap(j, k)
                if overlap_j > 0.5 or overlap_k > 0.5: # merge if overlap too much
                    self.merge_regions(j, k)
            ## evaluate
            iter_metrics = self.eval_print_result(i)
            all_metrics.append(iter_metrics)
        return np.array(all_metrics)
    
class BoundaryLSE_Exploit(BoundaryLSE):

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            None,
            kernel=self.kernel,
            exploit=True
        )
        self.region_data_selection(region_idx)
    
    def merge_regions(self, i, j):
        return super().merge_regions(i, j, True)

class Boundary_PureExploit(BoundaryLSE):

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            None,
            kernel=self.kernel,
            exploit="full"
        )
        self.region_data_selection(region_idx)

    def merge_regions(self, i, j):
        return super().merge_regions(i, j, "full")

class Boundary_StrongExploit(BoundaryLSE):

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            None,
            kernel=self.kernel,
            exploit="strong"
        )
        self.region_data_selection(region_idx)

    def merge_regions(self, i, j):
        return super().merge_regions(i, j, "strong")

class Boundary_LocalRandom(BoundaryLSE):

    def init_region(
            self, 
            centroid, 
            region_idx
        ):
        # initialize region
        self.regions[region_idx] = TrustRegion(
            self.h,
            self.epsilon,
            None,
            None,
            self.v_init,
            self.d,
            centroid,
            self.n_uniform,
            self.device,
            self.v_max,
            self.lengthscales_prior,
            None,
            kernel=self.kernel,
            exploit="random"
        )
        self.region_data_selection(region_idx)

    def merge_regions(self, i, j):
        return super().merge_regions(i, j, "random")
    
class TrustRegion_1(TrustRegion):
    # Randomly choose the new centroid from the pool
    def update_centroid(self, lse_preds, y_mean, X_sample):
        # random index
        random_idx = np.random.choice(X_sample.shape[0], 1)[0]
        new_centroid = X_sample[random_idx, :]
        return new_centroid
    
class TrustRegion_2(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*4/(1 + torch.exp(
            -6 + 8*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        ))
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
class TrustRegion_3(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*2/(1 + torch.exp(
            -9 + 12*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        ))
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
class TrustRegion_4(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*2/(1 + torch.exp(
            -3 + 4*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        ))
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
class TrustRegion_5(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*(
            4 - 4*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        )
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume

class TrustRegion_6(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*(
            3.25 - 3*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        )
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
class TrustRegion_7(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        region_lcb, min_idx = torch.min(lcb, 0)
        region_ucb, max_idx = torch.max(ucb, 0)
        region_mean = region_lcb/2 + region_ucb/2
        region_std = (region_ucb - region_lcb)/(2*BETA)
        new_volume = self.volume*(
            2.25 - 2*Normal(0, 1).cdf((torch.abs(region_mean - self.h)/region_std))
        )
        new_volume = min(new_volume.detach().cpu().item(), self.v_max)
        return new_volume
    
class TrustRegion_8(TrustRegion):
    def update_volumne(self, lcb, ucb, y_std):
        return self.volume
    
class BoundaryLSE_TRUpdate(BoundaryLSE):
    def __init__(
        self, 
        h=0, 
        epsilon=0.01, 
        n_regions=10, 
        d=10, 
        v_init=1e-10, 
        batch_size=10, 
        acq_name="straddle", 
        n_eval=10000, 
        v_max=0.01, 
        lengthscales_prior=None, 
        kernel="matern",
        tr_update_idx=1
    ):
        super().__init__(
            h, epsilon, n_regions, d, v_init, 
            batch_size, acq_name, n_eval, v_max, 
            lengthscales_prior, kernel
        )
        self.tr_update_idx = tr_update_idx
    
    def init_region(self, centroid, region_idx):
        # initialize region
        trust_region_class = f"TrustRegion_{self.tr_update_idx}"
        exec(f"self.regions[region_idx] = {trust_region_class}(" +
            "self.h, self.epsilon, None, None, self.v_init, self.d, centroid, " +
            "self.n_uniform, self.device, self.v_max, self.lengthscales_prior," +
            "None, kernel=self.kernel)")
        self.region_data_selection(region_idx)