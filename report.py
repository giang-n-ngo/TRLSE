import sys
import os

RUNS = 10

def check_results(result_root_path):
    check_file_path = f"{result_root_path}/f1_super.txt"
    if not os.path.exists(check_file_path):
        return RUNS
    else:
        with open(check_file_path, "rb") as f:
            num_lines = sum(1 for _ in f)
            return RUNS - num_lines

if __name__ == "__main__":
    sys.stdout = open("report.txt", "w")
    # Check main results
    print("Main results")
    print("- Random")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/random/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("- Straddle")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/straddle/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("- TRLSE")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/TR_straddle/boundary/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("===========================")
    # Check Ablation
    print("Ablation study")
    print("- V_init")
    for data_name in ["levy10D_v_init_2", "levy10D_v_init_4", "levy10D_v_init_10", "levy10D_v_init_20",
                      "levy100D_v_init_10", "levy100D_v_init_20", "levy100D_v_init_40", "levy100D_v_init_60"]:
        result_root_path = f"results/{data_name}/TR_straddle/boundary/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("- C")
    for data_name in ["levy10D_C_5", "levy10D_C_20", "levy10D_C_60",
                      "levy100D_C_5", "levy100D_C_20", "levy100D_C_60", "levy100D_C_100"]:
        result_root_path = f"results/{data_name}/TR_straddle/boundary/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("- Random init")
    for data_name in ["levy10D", "levy100D"]:
        result_root_path = f"results/{data_name}/TR_straddle/boundary_random/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("- OneGP")
    for data_name in ["levy10D", "levy100D"]:
        result_root_path = f"results/{data_name}/TR_straddle/onegp/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name}: {remaining_runs}")
    print("===========================")
    # Check appendix
    print("- C2LSE")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/c2lse/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + C2LSE: {remaining_runs}")
        result_root_path = f"results/{data_name}/TR_c2lse/boundary/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + TR_C2LSE: {remaining_runs}")
    print("- TS")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/TS/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + TS: {remaining_runs}")
        result_root_path = f"results/{data_name}/TR_TS/boundary/matern"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + TR_TS: {remaining_runs}")
    print("===========================")
    # Check kernel
    print("- RBF")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/random/rbf"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + random: {remaining_runs}")
        result_root_path = f"results/{data_name}/straddle/rbf"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + straddle: {remaining_runs}")
        result_root_path = f"results/{data_name}/TR_straddle/boundary/rbf"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + TR_straddle: {remaining_runs}")
    print("- RQ")
    for data_name in ["levy10D", "AA33D", "Mazda74D", \
                      "levy100D", "Vehicle124", "ackley200D", \
                      "MC2D", "mishra03", \
                      "rosenbrock1000D", "trid1000D"]:
        result_root_path = f"results/{data_name}/random/rq"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + random: {remaining_runs}")
        result_root_path = f"results/{data_name}/straddle/rq"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + straddle: {remaining_runs}")
        result_root_path = f"results/{data_name}/TR_straddle/boundary/rq"
        remaining_runs = check_results(result_root_path)
        print(f"  - {data_name} + TR_straddle: {remaining_runs}")
    sys.stdout.close()