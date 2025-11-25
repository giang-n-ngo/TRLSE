import gpytorch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, RQKernel
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

RUNS = 10

def init_kernel(dim, kernel="matern"):
    if dim > 200:
        lengthscale_constraint = Interval(0.5, 4.0)
        lengthscale_prior = LogNormalPrior(math.sqrt(2) + math.log(dim)/2, math.sqrt(3))
    else:
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

def fit_gp_model(X, y, device, dtype, kernel="matern"):
    gp_model = SingleTaskGP(
        X.detach().to(device, dtype), 
        y.detach().to(device, dtype), 
        covar_module=init_kernel(X.shape[1], kernel),
        likelihood=GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)),
    )
    gp_model = gp_model.to(device, dtype=dtype)
    if X.shape[0]>0:
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        mll = mll.to(device)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            fit_gpytorch_mll(mll)
    gp_model.eval()
    return gp_model

def batch_mean_std(X, gp_model):
    # for maximizing memory
    max_batch = 100
    gp_model.eval()
    gp_model = gp_model.to(X)
    y_mean = []
    y_std = []
    with torch.no_grad():
        for i in range(X.shape[0]//max_batch + 1):
            X_slice = X[(i*max_batch):((i+1)*max_batch), :].to(X)
            if X_slice.shape[0] > 0:
                f_slice_preds = gp_model.posterior(X_slice)
                y_mean.append(f_slice_preds.mean)
                y_std.append(f_slice_preds.variance**0.5)
    y_mean = torch.cat(y_mean, dim=0).to(X)
    y_std = torch.cat(y_std, dim=0).to(X)
    return y_mean, y_std

def plot_metrics(metrics, data, note, savefig=True):
    fig = plt.figure(figsize=(10, 5))
    iterations = [i for i in range(metrics.shape[0])]
    plt.plot(iterations, metrics, label="f1")
    plt.legend()
    if savefig:
        plt.savefig(f"results/{data}_{note}.png")
    plt.close(fig)

def check_synthetic(func_name, grid=True):
    from data_store import params
    from dataloaders import DataLoader
    func_params = params[func_name]
    loader = DataLoader(func_params, grid)
    X_test, y_test = loader.Xtest, loader.ytest
    if X_test.shape[1]==2:
        plt.scatter(X_test[:,0], X_test[:, 1], c=y_test)
        plt.show()
    print(f"Superlevel set percentage: {round((torch.sum(y_test>loader.h)*100/y_test.shape[0]).item(), 2)}%")
    q = []
    if func_params["superlevelset"]:
        imp_h = 0.8
    else:
        imp_h = 0.2
    y_quantile = [0]*20
    for i in range(100):
        _, y = loader.gen_observations_raw(10000)
        q.append(y.quantile(imp_h))
        for i in range(20):
            y_quantile[i] += y.quantile(i*0.05)
    print("Top 20%: ", sum(q)/100)
    y_quantile = [item/100 for item in y_quantile]
    ytest = loader.ytest
    for i in range(20):
        print(f"{i*5}%", ytest.quantile(i*0.05), y_quantile[i])

def plot_region_data(regions, v_init, h):
    n_regions = len(regions)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax1 = ax.twinx()
    data = []
    volume = []
    for i in range(n_regions):
        region = regions[i]
        region_y = region.ytrain.detach().cpu().squeeze(1).numpy()
        data.append(region_y)
        ax.scatter([i+1]*region_y.shape[0], region_y, c="red")
        volume.append(np.log10(region.volume/v_init))
    ax1.bar([i+1 for i in range(n_regions)], volume, alpha=0.2)
    # ax.boxplot(data)
    ax.hlines(h, 1, n_regions)
    ax.set_xlabel("Region")
    ax.set_ylabel("y")
    plt.xticks([i+1 for i in range(n_regions)], [str(i+1) for i in range(n_regions)])
    ax1.set_ylabel("Region volume (log 10)")
    plt.show()

def load_or_gen_initial_data(dataloader, n_regions, root_path, run_idx):
    X_path = f"{root_path}/{run_idx}_X.pt"
    y_path = f"{root_path}/{run_idx}_y.pt"
    if os.path.exists(X_path) and os.path.exists(y_path):
        X_init = torch.load(X_path)
        y_init = torch.load(y_path)
    else:
        X_init, y_init = dataloader.gen_observations(n_regions)
        os.makedirs(root_path, exist_ok=True)
        torch.save(X_init, X_path)
        torch.save(y_init, y_path)
    return X_init, y_init

def check_results(result_root_path):
    check_file_path = f"{result_root_path}/f1_super.txt"
    if not os.path.exists(check_file_path):
        return RUNS
    else:
        with open(check_file_path, "rb") as f:
            num_lines = sum(1 for _ in f)
            return RUNS - num_lines