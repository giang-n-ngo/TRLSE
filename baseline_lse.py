from baseline_models import BaseContinuousModel, RandomeSampling
from dataloaders import DataLoader
from data_store import params
from utils import load_or_gen_initial_data, check_results
import argparse
import os
import torch
import warnings
warnings.filterwarnings("ignore")

RUNS = 10

def save_metrics(metrics, result_root_path):
    for i, col in enumerate([
        "f1_sub", "prec_sub", "rec_sub", 
        "f1_super", "prec_super", "rec_super",
        "distance"
    ]):
        with open(f"{result_root_path}/{col}.txt", "a+") as f:
            f.write(f"{','.join([str(item) for item in metrics[:, i].tolist()])}\n")

def exp(data_name, acq_name, kernel, run_idx):
    os.makedirs(f"results/{data_name}/{acq_name}/{kernel}", exist_ok=True)
    os.makedirs(f"data/{data_name}", exist_ok=True)
    data_root_path = f"data/{data_name}"
    func_params = params[data_name]
    if func_params["grid_size"] is None:
        grid = False
    else:
        grid = True
    dataloader = DataLoader(func_params, grid)
    X_init, y_init = load_or_gen_initial_data(
        dataloader, 
        func_params["n_regions"], 
        data_root_path, 
        run_idx
    )
    if acq_name == "random":
        lse_model = RandomeSampling(
            superlevelset=func_params["superlevelset"],
            data_container=dataloader,
            Xtrain=X_init,
            ytrain=y_init,
            acq_name=acq_name,
            epsilon=func_params["epsilon"],
            beta=1.96,
            kernel=kernel
        )
    else:
        lse_model = BaseContinuousModel(
            superlevelset=func_params["superlevelset"],
            data_container=dataloader,
            Xtrain=X_init,
            ytrain=y_init,
            acq_name=acq_name,
            epsilon=func_params["epsilon"],
            beta=1.96,
            kernel=kernel
        )
    metrics = lse_model.run(func_params["budget"])
    return metrics, lse_model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="MC2D")
    parser.add_argument("--acq_name", type=str, default="straddle")
    parser.add_argument("--kernel", type=str, default="matern")
    parser.add_argument("--save", action="store_true", help="Whether to save results")
    args = parser.parse_args()
    print(args.data_name)
    result_root_path = f"results/{args.data_name}/{args.acq_name}/{args.kernel}"
    remaining_runs = check_results(result_root_path)
    # for i in range(RUNS - remaining_runs, RUNS):
    for i in range(RUNS):
        print(f"Run {i}")
        metrics, lse_model = exp(args.data_name, args.acq_name, args.kernel, i)
        # save GP model
        model_path = f"{result_root_path}/model_run_{i}.pth"
        torch.save(lse_model.GP_model.state_dict(), model_path)
        del lse_model
        # if args.save:
        #     save_metrics(metrics, result_root_path)