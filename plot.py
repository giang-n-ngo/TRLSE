import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from data_store import params

label_name = {
    "straddle": "Straddle",
    "c2lse": "C2LSE",
    "TS": "TS"
}

locs = {
    "levy10D": "upper left",
    "alpine10D": "lower right",
    "ackley10D": "lower right",
    "Protein20D": "lower right",
    "AA33D": "lower right",
    "levy100D": "upper left",
    "rosen100D": "upper left",
    "ackley200D": "center right",
    "Mazda74D": "lower right",
    "MC2D": "lower right",
    "mishra03": "lower right",
    "ackley2D": "lower right",
    "Vehicle124": "upper left",
    "ackley400D": "upper left",
    "sphere1000D": "upper left",
    "rosenbrock1000D": "center right",
    "trid1000D": "center right"
}

def load_results_from_drive(filename):
    try:
        if not os.path.exists(filename):
            print(f"Error: File not found at {filename}")
            return None
        with open(filename, 'r') as f:
            results = [list(map(float, line.strip().split(','))) for line in f][-10:]
        return np.array(results)  # Return the file content as a string

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_initial_results(filename):
    initial_results = load_results_from_drive(filename)
    return initial_results

def read_results(filename):
    """Reads the specified file using load_results_from_drive."""
    result = load_results_from_drive(filename)
    return result

def load_results(data_name, acq_name, kernel="", TR=False, random=False, metric="f1", concat_initial=False, onegp=False, boundary_label="", tr_update_idx=0):
    TR_prefix = "TR_" if TR else ""
    boundary = "/boundary" if TR else ""
    boundary = "/boundary_random" if random else boundary
    boundary = "/onegp" if onegp else boundary
    boundary = f"/{boundary_label}" if boundary_label!="" else boundary
    boundary = f"/boundary_tr_update_{tr_update_idx}" if tr_update_idx>0 else boundary
    if kernel != "":
        kernel = f"{kernel}/"
    if metric == "f1":
        result = read_results(f"results/{data_name}/{TR_prefix}{acq_name}{boundary}/{kernel}f1_super.txt")
    elif metric == "prec":
        result = read_results(f"results/{data_name}/{TR_prefix}{acq_name}{boundary}/{kernel}prec_super.txt")
    elif metric == "rec":
        result = read_results(f"results/{data_name}/{TR_prefix}{acq_name}{boundary}/{kernel}rec_super.txt")
    if concat_initial:
        initial_results = load_initial_results(f"results/{data_name}/{TR_prefix}{acq_name}{boundary}/{kernel}initial_f1_super.txt")
        result = np.concatenate((initial_results[:result.shape[0]], result), axis=1)
    return result

def load_samples(data_name, acq_name="straddle", kernel="", HLSE=False, random=False, concat_initial=False, onegp=False, boundary_label="", tr_update_idx=0):
    if kernel != "":
        kernel = f"{kernel}/"    
    if HLSE:
        samples = read_results(f"results/{data_name}/HLSE/samples.txt")
    else:
        if random:
            samples = read_results(f"results/{data_name}/TR_{acq_name}/boundary_random/{kernel}samples.txt")
        elif onegp:
            samples = read_results(f"results/{data_name}/TR_{acq_name}/onegp/{kernel}samples.txt")
        elif tr_update_idx>0:
            samples = read_results(f"results/{data_name}/TR_{acq_name}/boundary_tr_update_{tr_update_idx}/{kernel}samples.txt")
        elif boundary_label=="":
            samples = read_results(f"results/{data_name}/TR_{acq_name}/boundary/{kernel}samples.txt")
        else:
            samples = read_results(f"results/{data_name}/TR_{acq_name}/{boundary_label}/{kernel}samples.txt")
    if concat_initial:
        initial_samples = np.array([[params[data_name]["n_regions"]]*len(samples)]).T
        samples = np.concatenate((initial_samples, samples), axis=1)
    return samples

def prepare_series(all_results):
    """
    Prepare the series for plotting by calculating median and interquartile range.
    all_results is of shape (10, num_iter) and is a numpy array
    """
    median_result = np.median(all_results, axis=0)
    q1_result = np.percentile(all_results, 25, axis=0)
    q3_result = np.percentile(all_results, 75, axis=0)
    return median_result, q1_result, q3_result

def plot_baselines(data_name, acq_name, kernel="matern", random=False, legend=True, ylabel=True, save_path=""):
    budget = params[data_name]["budget"]
    plt.figure(figsize=(5, 2.5))
    if random:
    # plot random
        random_f1 = load_results(data_name, "random", kernel=kernel, TR=False)
        mean_random_f1, min_random_f1, max_random_f1 = prepare_series(random_f1)
        random_start = params[data_name]["n_regions"]
        x_random = [i+1 for i in np.linspace(random_start, budget, mean_random_f1.shape[0])]
        mean_random_f1 = mean_random_f1[:len(x_random)]
        plt.plot(x_random, mean_random_f1, color="green", label=acq_name)
        plt.fill_between(x_random, min_random_f1, max_random_f1, color="lightgreen", alpha=0.5)
    # plot simple
    simple_f1 = load_results(data_name, acq_name, kernel=kernel, TR=False, concat_initial=False)
    mean_simple_f1, min_simple_f1, max_simple_f1 = prepare_series(simple_f1)
    x_simple = [i+1 for i in np.linspace(random_start, budget, mean_simple_f1.shape[0])]
    mean_simple_f1 = mean_simple_f1[:len(x_simple)]
    plt.plot(x_simple, mean_simple_f1, color="b", label=acq_name)
    plt.fill_between(x_simple, min_simple_f1, max_simple_f1, color="lightskyblue", alpha=0.5)
    # plot HLSE
    hlse_f1 = load_results(data_name, "HLSE", kernel="", TR=False)
    mean_hlse_f1, min_hlse_f1, max_hlse_f1 = prepare_series(hlse_f1)
    hlse_samples = load_samples(data_name, HLSE=True)[0]
    plt.plot(hlse_samples, mean_hlse_f1, color="orange", label="HLSE")
    plt.fill_between(hlse_samples, min_hlse_f1, max_hlse_f1, color="lightsalmon", alpha=0.5)
    # plot TRLSE
    starting_point = params[data_name]["n_regions"]
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False)
    t_common = np.linspace(starting_point, budget, 100)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    )
    mean_combined, min_combined, max_combined = prepare_series(interp_series)
    plt.plot(t_common, mean_combined, color="red", label=f"TRLSE+{label_name[acq_name]}")
    plt.fill_between(t_common, min_combined, max_combined, color="pink", alpha=0.5)
    if random:
        legend_elements = [
            Line2D([0], [0], color='green', lw=1.5, label="Random"),
            Line2D([0], [0], color='b', lw=1.5, label=label_name[acq_name]),
            Line2D([0], [0], color='r', lw=1.5, label=f"TRLSE"),
            Line2D([0], [0], color='orange', lw=1.5, label='HLSE')
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color='b', lw=1.5, label=label_name[acq_name]),
            Line2D([0], [0], color='r', lw=1.5, label=f"TRLSE"),
            Line2D([0], [0], color='orange', lw=1.5, label='HLSE')
        ]        
    if legend:
        plt.legend(handles=legend_elements, loc=locs[data_name])
    plt.xlabel("#function evaluations")
    if ylabel:
        plt.ylabel("F1-score")
    random_suffix = "_random" if random else ""
    if save_path!="":
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"plot_figures/{acq_name}_{data_name}_{kernel}{random_suffix}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_main(data_name):
    plot_baselines(data_name, "straddle", "matern", random=True)

def plot_random(data_name, ylabel=True, legend=True):
    acq_name = "straddle"
    kernel = "matern"
    starting_point = params[data_name]["n_regions"]
    budget = params[data_name]["budget"]
    plt.figure(figsize=(5, 2.5))
    # plot onegp
    TR_onegp_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=False, onegp=True)
    onegp_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=False, onegp=True)
    t_common = np.linspace(starting_point, budget, 100)  # Define a common timeline
    onegp_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(onegp_samples, TR_onegp_f1)]
    )
    mean_onegp_combined, min_onegp_combined, max_onegp_combined = prepare_series(onegp_interp_series)
    plt.plot(t_common, mean_onegp_combined, color="green", label=f"TRLSE with only global GP")
    plt.fill_between(t_common, min_onegp_combined, max_onegp_combined, color="lightgreen", alpha=0.5)
    # plot TRLSE
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False)
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    )
    mean_combined, min_combined, max_combined = prepare_series(interp_series)
    plt.plot(t_common, mean_combined, color="red", label="TRLSE")
    plt.fill_between(t_common, min_combined, max_combined, color="pink", alpha=0.5)
    # plot random
    TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=True, concat_initial=False)
    random_samples = load_samples(data_name, acq_name, kernel=kernel, random=True, concat_initial=False)
    random_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(random_samples, TR_random_f1)]
    )
    mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    plt.plot(t_common, mean_random_combined, color="blue", label=f"TRLSE with random TRs")
    plt.fill_between(t_common, min_random_combined, max_random_combined, color="lightskyblue", alpha=0.5)
    if legend:
        legend_elements = [
            Line2D([0], [0], color="red", lw=1.5, label="TRLSE"),
            Line2D([0], [0], color="b", lw=1.5, label="TRLSE w/ random TRs"),
            Line2D([0], [0], color="green", lw=1.5, label="TRLSE w/ only global GP")
        ]
        legend_loc = "lower right" if data_name == "levy10D" else locs[data_name]
        plt.legend(handles=legend_elements, loc=legend_loc)
    plt.xlabel("#function evaluations")
    if ylabel:
        plt.ylabel("F1-score")
    plt.savefig(f"plot_figures/random_onegp_{data_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_ablation(data_name, ablation, ylabel=True):
    legend_color = ["r", "b", "g", "orange", "olive"]
    legend_std_color = ["pink", "lightskyblue", "lightgreen", "lightsalmon", "darkkhaki"]
    ablation_name = {
        "C": "n_regions",
        "v_init": "v_init"
    }
    val_set = {
        "C": {
            "levy10D": [5, 20, 60],
            "levy100D": [5, 20, 60, 100]
        },
        "v_init": {
            "levy10D": [2, 4, 10, 20],
            "levy100D": [10, 20, 40, 60]
        }
    }
    acq_name = "straddle"
    kernel = "matern"
    budget = params[data_name]["budget"]
    starting_point = params[data_name]["n_regions"]
    t_common = np.linspace(starting_point, budget, 100)
    if ablation=="v_init" and data_name=="levy10D":
        first_label_str = "1e-5"
    else:
        first_label_str = f"{params[data_name][ablation_name[ablation]]}"
    legend_elements = {params[data_name][ablation_name[ablation]]: Line2D([0], [0], color=legend_color[0], lw=1.5, label=first_label_str)}
    plt.figure(figsize=(5, 2.5))
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False)
    interp_series = [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    mean_combined, min_combined, max_combined = prepare_series(np.asarray(interp_series))
    plt.plot(t_common, mean_combined, color=legend_color[0]) 
    plt.fill_between(t_common, min_combined, max_combined, color=legend_std_color[0], alpha=0.5)
    for i, val in enumerate(val_set[ablation][data_name]):
        val_data_name = f"{data_name}_{ablation}_{val}"
        if ablation=="C":
            starting_point = params[val_data_name]["n_regions"]
            t_common = np.linspace(starting_point, budget, 100)
        TR_f1 = load_results(val_data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
        samples = load_samples(val_data_name, acq_name, kernel=kernel, concat_initial=False)
        interp_series = [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                            for t, series in zip(samples, TR_f1)]
        mean_combined, min_combined, max_combined = prepare_series(np.asarray(interp_series))
        plt.plot(t_common, mean_combined, color=legend_color[i+1])
        plt.fill_between(t_common, min_combined, max_combined, color=legend_std_color[i+1], alpha=0.5)
        if ablation=="v_init":
            # if val<=4:
            #     val_str = f"1e-{val}"
            # else:
            #     val = 10**(-val)
            #     val_str = str(val)
            val_str = f"1e-{val}"
            val = 10**(-val)
        else:
            val_str = str(val)
        legend_elements[val] = Line2D([0], [0], color=legend_color[i+1], lw=1.5, label=f"{val_str}")
    legend_elements = [value for _, value in sorted(legend_elements.items())]
    if ablation=="v_init" and data_name=="levy10D":
        legend_loc = "lower right"
    elif ablation=="C" and data_name=="levy10D":
        legend_loc = "lower right"
    else:
        legend_loc = locs[data_name]
    plt.legend(handles=legend_elements, loc=legend_loc)
    plt.xlabel("#function evaluations")
    if ylabel:
        plt.ylabel("F1-score")
    plt.savefig(f"plot_figures/ablation_{ablation}_{data_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_prec_rec():
    data_name = "levy100D"
    acq_name = "straddle"
    kernel = "matern"
    budget = params[data_name]["budget"]
    fig, ax1 = plt.subplots(figsize=(4, 3))
    ax2 = ax1.twinx()
    simple_precision = load_results(data_name, acq_name, kernel=kernel, TR=False, metric="prec")
    simple_recall = load_results(data_name, acq_name, kernel=kernel, TR=False, metric="rec")
    # plot simple
    x_simple = [i+1 for i in range(0, budget, 10*params[data_name]["batch_size"])]
    ax1.plot(x_simple, simple_precision.mean(0), color="b", label=f"{acq_name} - precision")
    ax2.plot(x_simple, simple_recall.mean(0), color="r", label=f"{acq_name} - recall")
    legend_elements = [
        Line2D([0], [0], color='b', lw=1.5, label=f"Straddle - precision"),
        Line2D([0], [0], color='r', lw=1.5, label=f"Straddle - recall"),
    ]
    plt.legend(handles=legend_elements, loc="center right")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 0.01)
    ax1.set_xlabel("#function evaluations")
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Recall")
    plt.savefig("plot_figures/prec_rec.pdf",  dpi=300, bbox_inches="tight")

def plot_kernel(data_name, kernel, random):
    plot_baselines(data_name, "straddle", kernel, random)

def plot_onegp(data_name):
    acq_name = "straddle"
    kernel = "matern"
    budget = params[data_name]["budget"]
    plt.figure(figsize=(4, 3))
    starting_point = params[data_name]["n_regions"]
    # plot onegp
    TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=False, onegp=True)
    random_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=False, onegp=True)
    end_point = int(random_samples[:, -1].mean())
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    random_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(random_samples, TR_random_f1)]
    )
    mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    plt.plot(t_common, mean_random_combined, color="blue", label=f"TRLSE with only global GP")
    plt.fill_between(t_common, min_random_combined, max_random_combined, color="lightskyblue", alpha=0.5)
    # plot TRLSE
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False)
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    )
    mean_combined, min_combined, max_combined = prepare_series(interp_series)
    plt.plot(t_common, mean_combined, color="red", label="TRLSE")
    plt.fill_between(t_common, min_combined, max_combined, color="pink", alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color="red", lw=1.5, label="TRLSE"),
        Line2D([0], [0], color="b", lw=1.5, label="TRLSE with only global GP")
    ]
    legend_loc = "lower right" if data_name == "levy10D" else locs[data_name]
    plt.legend(handles=legend_elements, loc=legend_loc)
    plt.xlabel("#function evaluations")
    plt.ylabel("F1-score")
    plt.savefig(f"plot_figures/onegp_{data_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_beta(data_name):
    legend_color = ["r", "b", "g", "orange", "olive"]
    legend_std_color = ["pink", "lightskyblue", "lightgreen", "lightsalmon", "darkkhaki"]
    acq_name = "straddle"
    kernel = "matern"
    budget = params[data_name]["budget"]
    plt.figure(figsize=(4, 3))
    starting_point = params[data_name]["n_regions"]
    # plot pure exploit
    TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=True, boundary_label="pure_exploit")
    random_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=True, boundary_label="pure_exploit")
    end_point = min(int(random_samples[:, -1].mean()), budget)
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    random_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(random_samples, TR_random_f1)]
    )
    mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    plt.plot(t_common, mean_random_combined, color=legend_color[-1], label=f"0")
    plt.fill_between(t_common, min_random_combined, max_random_combined, color=legend_std_color[-1], alpha=0.5)
    # plot strong exploit
    TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=True, boundary_label="strong_exploit")
    random_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=True, boundary_label="strong_exploit")
    end_point = min(int(random_samples[:, -1].mean()), budget)
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    random_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(random_samples, TR_random_f1)]
    )
    mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    plt.plot(t_common, mean_random_combined, color=legend_color[-2], label=f"0.49")
    plt.fill_between(t_common, min_random_combined, max_random_combined, color=legend_std_color[-2], alpha=0.5)
    # plot exploit
    TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=True, boundary_label="exploit")
    random_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=True, boundary_label="exploit")
    end_point = min(int(random_samples[:, -1].mean()), budget)
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    random_interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(random_samples, TR_random_f1)]
    )
    mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    plt.plot(t_common, mean_random_combined, color=legend_color[-3], label=f"0.98")
    plt.fill_between(t_common, min_random_combined, max_random_combined, color=legend_std_color[-3], alpha=0.5)
    # # plot random local
    # TR_random_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, random=False, concat_initial=True, boundary_label="local_random")
    # random_samples = load_samples(data_name, acq_name, kernel=kernel, random=False, concat_initial=True, boundary_label="local_random")
    # end_point = min(int(random_samples[:, -1].mean()), budget)
    # t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    # random_interp_series = np.asarray(
    #     [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
    #                     for t, series in zip(random_samples, TR_random_f1)]
    # )
    # mean_random_combined, min_random_combined, max_random_combined = prepare_series(random_interp_series)
    # plt.plot(t_common, mean_random_combined, color=legend_color[-4], label=f"random local sampling")
    # plt.fill_between(t_common, min_random_combined, max_random_combined, color=legend_std_color[-4], alpha=0.5)
    # plot TRLSE
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=True)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=True)
    t_common = np.linspace(starting_point, end_point, 100)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    )
    mean_combined, min_combined, max_combined = prepare_series(interp_series)
    plt.plot(t_common, mean_combined, color=legend_color[-5], label="1.96")
    plt.fill_between(t_common, min_combined, max_combined, color=legend_std_color[-5], alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color=legend_color[-5], lw=1.5, label="1.96"),
        Line2D([0], [0], color=legend_color[-3], lw=1.5, label="0.98"),
        Line2D([0], [0], color=legend_color[-2], lw=1.5, label="0.49"),
        Line2D([0], [0], color=legend_color[-1], lw=1.5, label="0"),
        # Line2D([0], [0], color=legend_color[-4], lw=1.5, label="random local sampling")
    ]
    plt.legend(handles=legend_elements, loc=locs[data_name])
    plt.xlabel("#function evaluations")
    plt.ylabel("F1-score")
    plt.savefig(f"plot_figures/exploit_{data_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_tr_update(data_name, tr_updates=[1], ylabel=True):
    legend_color = ["r", "b", "g", "orange", "olive"]
    legend_std_color = ["pink", "lightskyblue", "lightgreen", "lightsalmon", "darkkhaki"]
    acq_name = "straddle"
    kernel = "matern"
    value_mapping ={
        1: [0.24, 1.76],
        2: [0.48, 3.52],
        3: [0.09, 1.91],
        4: [0.54, 1.46],
        5: [0, 2],
        6: [0.25, 1.75],
        7: [0.25, 1.25]
    }
    budget = params[data_name]["budget"]
    plt.figure(figsize=(5, 2.5))
    # plot TRLSE
    starting_point = params[data_name]["n_regions"]
    TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False)
    samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False)
    t_common = np.linspace(starting_point, budget, 100)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, TR_f1)]
    )
    mean_combined, min_combined, max_combined = prepare_series(interp_series)
    plt.plot(t_common, mean_combined, color=legend_color[0])
    plt.fill_between(t_common, min_combined, max_combined, color=legend_std_color[0], alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color=legend_color[0], lw=1.5, label=rf"$S_1$:{value_mapping[1][0]}-{value_mapping[1][1]}")
    ]
    # plot other results
    for i, tr_update_idx in enumerate(tr_updates):
        TR_f1 = load_results(data_name, acq_name, kernel=kernel, TR=True, concat_initial=False, tr_update_idx=tr_update_idx)
        samples = load_samples(data_name, acq_name, kernel=kernel, concat_initial=False, tr_update_idx=tr_update_idx)
        t_common = np.linspace(starting_point, budget, 100)  # Define a common timeline
        interp_series = np.asarray(
            [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                            for t, series in zip(samples, TR_f1)]
        )
        mean_combined, min_combined, max_combined = prepare_series(interp_series)
        plt.plot(t_common, mean_combined, color=legend_color[i+1])
        plt.fill_between(t_common, min_combined, max_combined, color=legend_std_color[i+1], alpha=0.5)
        if tr_update_idx==8:
            label = r"$S(u)=1$"
        else:
            label = rf"$S_{tr_update_idx}$:{value_mapping[tr_update_idx][0]}-{value_mapping[tr_update_idx][1]}"
        legend_elements.append(Line2D([0], [0], color=legend_color[i+1], lw=1.5, label=label))

    legend_loc = "lower right"
    plt.legend(handles=legend_elements, loc=legend_loc)
    plt.xlabel("#function evaluations")
    if ylabel:
        plt.ylabel("F1-score")
    plt.savefig(f"plot_figures/{data_name}_tr_update_{"".join([str(item) for item in tr_updates])}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="MC2D")
    parser.add_argument("--acq_name", type=str, default="straddle")
    parser.add_argument("--kernel", type=str, default="matern")
    parser.add_argument("--section", type=str, default="main")
    parser.add_argument("--ablation", type=str, default="C")
    parser.add_argument("--random", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.section=="main":
        plot_main(args.data_name, args.random)
    elif args.section=="random":
        plot_random(args.data_name)
    elif args.section=="ablation":
        plot_ablation(args.data_name, args.ablation)
    elif args.section=="acq":
        plot_baselines(args.data_name, args.acq_name, "matern", args.random)
    elif args.section=="prec_rec":
        plot_prec_rec()
    elif args.section=="kernel":
        plot_kernel(args.data_name, args.kernel, args.random)
    elif args.section=="onegp":
        plot_onegp(args.data_name)
    elif args.section=="exploit":
        plot_beta(args.data_name)
    elif args.section=="tr_update":
        plot_tr_update(args.data_name, [2,3,4])
        plot_tr_update(args.data_name, [5,6,7])
        plot_tr_update(args.data_name, [8])