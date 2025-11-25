import os
import shutil

data_name_list = ["levy10D", "alpine10D", "ackley10D", 
                  "Protein20D", "AA33D", "Mazda74D",
                  "levy100D", "Vehicle124D", "ackley200D",
                  "rosenbrock1000D", "sphere1000D", "trid1000D",
                  "MC2D", "mishra03"]
acq_list = ["random", "straddle", "c2lse", "TS"]

for data_name in data_name_list:
    for acq in acq_list:
        folder_path = os.path.join("results", data_name, acq)
        shutil.rmtree(folder_path, ignore_errors=True)