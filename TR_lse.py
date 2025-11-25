from TR_models import BoundaryLSE, BoundaryLSE_RandomTR, BoundaryLSE_OneGP, BoundaryLSE_Exploit, Boundary_PureExploit, Boundary_StrongExploit, Boundary_LocalRandom, BoundaryLSE_TRUpdate
from dataloaders import DataLoader
from data_store import params
from utils import load_or_gen_initial_data, check_results
from matplotlib.lines import Line2D
import argparse
import os
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore")

RUNS = 10

def init_model(data_name, acq_name, run_idx, hdlse="boundary", kernel="matern", tr_update_idx=0):
    func_params = params[data_name]
    if func_params["grid_size"] is None:
        dataloader = DataLoader(func_params, grid=False)
    else:
        dataloader = DataLoader(func_params)
    kwargs = {
        "h": dataloader.h,
        "epsilon": func_params["epsilon"],
        "n_regions": func_params["n_regions"],
        "d": func_params["n_dims"],
        "v_init": func_params["v_init"],
        "batch_size": func_params["batch_size"],
        "acq_name": acq_name,
        "v_max": func_params["v_max"],
        "kernel": kernel
    }
    if hdlse=="boundary":
        if tr_update_idx == 0:
            lse_model = BoundaryLSE(**kwargs)
        else:
            lse_model = BoundaryLSE_TRUpdate(**kwargs, tr_update_idx=tr_update_idx)
    elif hdlse=="boundary_random":
        lse_model = BoundaryLSE_RandomTR(**kwargs)
    elif hdlse=="onegp":
        lse_model = BoundaryLSE_OneGP(**kwargs)
    elif hdlse=="exploit":
        lse_model = BoundaryLSE_Exploit(**kwargs)
    elif hdlse=="strong_exploit":
        lse_model = Boundary_StrongExploit(**kwargs)
    elif hdlse=="pure_exploit":
        lse_model = Boundary_PureExploit(**kwargs)
    elif hdlse=="local_random":
        lse_model = Boundary_LocalRandom(**kwargs)
    if data_name in ["levy10D_v_init_2", "levy10D_v_init_4", "levy10D_v_init_10", "levy10D_v_init_20",
                    "levy100D_v_init_10", "levy100D_v_init_20", "levy100D_v_init_40", "levy100D_v_init_60"]:
        original_data_name = data_name.split("_")[0]
    else:
        original_data_name = data_name
    X_init, y_init = load_or_gen_initial_data(
        dataloader, 
        func_params["n_regions"], 
        f"data/{original_data_name}", 
        run_idx
    )
    lse_model.setup_model(X_init, y_init, dataloader.obj_func, dataloader)
    return lse_model, dataloader, func_params["n_iter"]

def save_metrics(metrics, result_root_path, note):
    for i, col in enumerate([
        "f1_sub", "prec_sub", "rec_sub", 
        "f1_super", "prec_super", "rec_super", 
        "vol", "samples", "restarts", "distance"
    ]):
        with open(f"{result_root_path}/{col}{note}.txt", "a+") as f:
            f.write(f"{','.join([str(item) for item in metrics[:, i].tolist()])}\n")

def plot_acq_val(result_root_path, run_idx, n_iter, local_acq_val, global_acq_val, global_local_acq_val=None):
    assert len(local_acq_val) == n_iter, "Length of local_acq_val does not match n_iter"
    assert len(global_acq_val) == n_iter, "Length of global_acq_val does not match n_iter"
    if global_local_acq_val is not None:
        assert len(global_local_acq_val) == n_iter, "Length of global_local_acq_val does not match n_iter"
    budget = len(local_acq_val)
    x = list(range(1, budget+1))
    plt.figure(figsize=(5, 4))
    plt.plot(x, local_acq_val, color="green")
    plt.plot(x, global_acq_val, color="orange")
    legend_elements = [Line2D([0], [0], color="green", lw=1.5, label="Local values")]
    legend_elements.append(Line2D([0], [0], color="orange", lw=1.5, label="Global values"))
    if global_local_acq_val is not None:
        plt.plot(x, global_local_acq_val, color="blue")
        legend_elements.append(Line2D([0], [0], color="blue", lw=1.5, 
                label="Global values at\nlocal points"))
    plt.legend(handles=legend_elements, loc="upper right")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition value")
    plt.savefig(f"{result_root_path}/{run_idx}_acq_val.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def exp(data_name, acq_name, run_idx, hdlse, kernel, tr_update_idx=0):
    if tr_update_idx == 0:
        tr_update = ""
    else:
        tr_update = f"_tr_update_{tr_update_idx}"
    result_root_path = f"results/{data_name}/TR_{acq_name}/{hdlse}{tr_update}/{kernel}"
    os.makedirs(result_root_path, exist_ok=True)
    if os.path.exists(result_root_path):
        print("Result directory setup correctly!")
    lse_model, dataloader, n_iter = init_model(data_name, acq_name, run_idx, hdlse, kernel, tr_update_idx)
    metrics = lse_model.run(n_iter)
    plot_acq_val(result_root_path, run_idx, n_iter, lse_model.local_acq_val, lse_model.global_acq_val, lse_model.global_local_acq_val)
    return metrics, lse_model, dataloader


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="MC2D")
    parser.add_argument("--acq_name", type=str, default="straddle")
    parser.add_argument("--hdlse", type=str, default="boundary")
    parser.add_argument("--kernel", type=str, default="matern")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--save", action="store_true", help="Whether to save results")
    parser.add_argument("--run_idx", type=int, default=-1)
    parser.add_argument("--tr_update_idx", type=int, default=0)
    args = parser.parse_args()
    print(args.data_name, args.acq_name, args.hdlse, args.kernel)
    if args.tr_update_idx == 0:
        tr_update = ""
    else:
        tr_update = f"_tr_update_{args.tr_update_idx}"
    result_root_path = f"results/{args.data_name}/TR_{args.acq_name}/{args.hdlse}{tr_update}/{args.kernel}"
    if args.run_idx == -1:
        remaining_runs = check_results(result_root_path)
        # for i in range(RUNS - remaining_runs, RUNS):
        for i in range(RUNS):
            print(f"Run {i}")
            metrics, lse_model, _ = exp(args.data_name, args.acq_name, i, args.hdlse, args.kernel, args.tr_update_idx)
            # save the GP model
            model_path = f"{result_root_path}/model_run_{i}.pt"
            torch.save(lse_model.global_gp_init.state_dict(), model_path)
            # if args.save:
            #     save_metrics(metrics, result_root_path, args.note)
    else:
        print(f"Run {args.run_idx}")
        metrics, lse_model, _ = exp(args.data_name, args.acq_name, args.run_idx, args.hdlse, args.kernel, args.tr_update_idx)
        # save the GP model
        model_path = f"{result_root_path}/model_run_{i}.pt"
        torch.save(lse_model.global_gp_init.state_dict(), model_path)
        # if args.save:
        #     save_metrics(metrics, result_root_path, args.note)