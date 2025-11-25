import numpy as np
import scikit_posthocs as sp
from scipy import stats
from scipy.interpolate import interp1d
from data_store import params
from plot import load_results, load_samples

ALPHA = 0.05  # Significance level for statistical tests

def get_trlse(data_name, raw=False):
    acq_name = "straddle"
    kernel = "matern"
    results = load_results(data_name, acq_name, kernel, TR=True, random=False)
    if raw:
        return results
    samples = load_samples(data_name, acq_name, kernel=kernel)
    starting_point = params[data_name]["n_regions"]
    budget = params[data_name]["budget"]
    t_common = np.linspace(starting_point, budget, 100)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, results)]
    )
    return interp_series

def run_statistical_analysis(
        dataset_name, 
        algorithms,
        algorithms_results,
        agg="mean"
    ):
    """
    Performs the full statistical analysis for a single dataset.
    1. Aggregates results from files.
    2. Runs the Friedman test.
    3. If significant, runs post-hoc Wilcoxon tests with Holm-Bonferroni correction.
    """
    print("-" * 70)
    print(f"📊 Analyzing Dataset: {dataset_name}")
    print("-" * 70)

    # --- Step 1: Aggregate Results ---
    # Load the 10 averaged scores for each algorithm into a dictionary.
    aggregated_scores = {}
    for i, algo in enumerate(algorithms):
        scores = algorithms_results[i]
        # scores should a 2D matrix with rows being 10 runs and columns being iterations
        if scores is not None:
            if agg == "mean":
                # average over iterations
                aggregated_scores[algo] = np.mean(scores, axis=1)
            elif agg == "last":
                aggregated_scores[algo] = scores[:, -1]
            assert len(aggregated_scores[algo]) == 10, f"Expected 10 runs for {algo}, got {len(aggregated_scores[algo])}."

    # Check if we have enough data to compare
    if len(aggregated_scores) < 2:
        print("Not enough valid algorithm results to perform a statistical test.\n")
        return

    # Create a 2D NumPy array from the aggregated scores for the tests.
    # Shape: (num_runs, num_algorithms) -> e.g., (10, 4)
    data = np.array(list(aggregated_scores.values())).T
    
    # --- Step 2: Friedman Test ---
    # This test checks if there are any significant differences among the algorithm groups.
    try:
        friedman_stat, p_friedman = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
        
        print(f"🔬 Friedman Test Results:")
        print(f"   - Statistic: {friedman_stat:.4f}")
        print(f"   - p-value: {p_friedman:.4f}")

        # --- Step 3: Post-Hoc Tests (if Friedman was significant) ---
        if p_friedman < ALPHA:
            print(f"\n✅ The Friedman test is significant (p < {ALPHA}).")
            print("   Running post-hoc tests to find specific differences...\n")
            
            # Perform pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
            # The result is a DataFrame where each cell is the p-value for the comparison
            # between the row and column algorithms.
            p_values_posthoc = sp.posthoc_wilcoxon(data, p_adjust='holm')
            
            # Set algorithm names as index and columns for readability
            p_values_posthoc.columns = aggregated_scores.keys()
            p_values_posthoc.index = aggregated_scores.keys()
            
            print("Pairwise Wilcoxon Test Results (Holm-Bonferroni corrected p-values):")
            print(p_values_posthoc.round(4))
            
            # Interpretation help
            print("\n💡 Interpretation:")
            my_algo_name = algorithms[0]
            for other_algo in algorithms[1:]:
                if other_algo in p_values_posthoc.index:
                    p_val = p_values_posthoc.loc[my_algo_name, other_algo]
                    if p_val < ALPHA:
                        print(f"   - {my_algo_name} is SIGNIFICANTLY BETTER than {other_algo} (p={p_val:.4f})")
                    else:
                        print(f"   - {my_algo_name} is NOT significantly better than {other_algo} (p={p_val:.4f})")

        else:
            print(f"\n❌ The Friedman test is not significant (p >= {ALPHA}).")
            print("   There is no statistical evidence of a difference among the algorithms.\n")

    except ValueError as e:
        print(f"An error occurred during the statistical test: {e}")
    
    print("\n")

def dataset_test(
        trlse_results, 
        random_results,
        simple_results,
        hlse_results,
        data_name, 
        hlse=False):
    n_iter = random_results.shape[1]
    samples = load_samples(data_name, "straddle", kernel="matern")
    if hlse:
        hlse_samples = load_samples(data_name, HLSE=True)[0]
        starting_point = hlse_samples[0]
        end_point = hlse_samples[-1]
    else:
        starting_point = params[data_name]["n_regions"]
        end_point = params[data_name]["budget"]
    t_common = np.linspace(starting_point, end_point, n_iter)  # Define a common timeline
    interp_series = np.asarray(
        [interp1d(t, series, kind='linear', fill_value='extrapolate')(t_common)
                        for t, series in zip(samples, trlse_results)]
    )
    print("Running with mean aggregation")
    run_statistical_analysis(
        data_name,
        ["TRLSE", "Random", "Straddle", "HLSE"],
        [trlse_results, random_results, simple_results, hlse_results],
        "mean"
    )
    print("Running with last aggregation")
    run_statistical_analysis(
        data_name,
        ["TRLSE", "Random", "Straddle", "HLSE"],
        [trlse_results, random_results, simple_results, hlse_results],
        "last"
    )


def t_test(data_name):
    print(data_name)
    trlse_results = get_trlse(data_name, raw=True)
    hlse_results = load_results(data_name, "HLSE", kernel="", TR=False)
    simple_results = load_results(data_name, "straddle", kernel="matern", TR=False)
    random_results = load_results(data_name, "random", kernel="matern", TR=False)
    random_pvalue = run_t_test(trlse_results, random_results, data_name)
    simple_pvalue = run_t_test(trlse_results, simple_results, data_name)
    hlse_pvalue = run_t_test(trlse_results, hlse_results, data_name)

if __name__=="__main__":
    for data_name in ["levy10D", "AA33D", "Mazda74D", "levy100D", "Vehicle124", "ackley200D", "trid1000D", "rosenbrock1000D"]:
        t_test(data_name)