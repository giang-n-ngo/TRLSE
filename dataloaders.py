import numpy as np
import pickle
import pandas as pd
import torch
import os
from collections import Counter
from itertools import product
from torch.quasirandom import SobolEngine
from sklearn.ensemble import RandomForestRegressor

def norm(X, X_range):
    normed_X = np.zeros(X.shape)
    if len(X_range)>1:
        for dim in X_range.keys():
            if X_range[dim][0]!=X_range[dim][1]:
                normed_X[..., dim] = (X[..., dim] - X_range[dim][0])/((X_range[dim][1] - X_range[dim][0]))
            else:
                normed_X[..., dim] = 0
    else:
        normed_X = (X - X_range[0][0])/((X_range[0][1] - X_range[0][0]))
    return torch.from_numpy(normed_X)

def denorm(normed_X, X_range):
    X = torch.zeros(normed_X.shape)
    if len(X_range)>1:
        for dim in X_range.keys():
            X[..., dim] = normed_X[..., dim]*(X_range[dim][1] - X_range[dim][0]) + X_range[dim][0]
    else:
        X = normed_X*(X_range[0][1] - X_range[0][0]) + X_range[0][0]
    return X

def standardize(X, sample_mean, sample_std):
    return (X-sample_mean)/sample_std

def min_max_scaling(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def fit_and_save_model(X, y, path):
    model = RandomForestRegressor().fit(X, y)
    preds = model.predict(X)
    print("MSE: ", ((preds - y)**2).mean())
    with open(path, "wb") as f:
        pickle.dump(model, f)

def save_X(X, path):
    X = torch.tensor(X)
    torch.save(X, path)

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    def processed_model(X):
        out = model.predict(X.detach().cpu().numpy())
        return torch.from_numpy(out)
    return processed_model

def data_generate_Protein(filename):
    
    data_all_pd = pd.read_csv(filename)
    whole_data = data_all_pd.values
    X_data_seq = whole_data[:, 0]
    Y_data = whole_data[:, 1]
    Y_data = Y_data[..., np.newaxis]
    
    # Get all the possible characters in the sequence
    char_all = []
    for item in X_data_seq:
        char_all += item
    char_distinct = set(char_all)
    
    # Convert the sequence in X_data to bags of words
    # The new input data will consist of the number of distinct characters and 
    # the corresponding 
    N = len(Y_data)
    n = len(char_distinct)
    X_data = np.zeros((N, n))
    for X_idx, item in enumerate(X_data_seq):
        counter = Counter(item)
        for idx, char in enumerate(char_distinct):
            X_data[X_idx, idx] = counter[char]
    return X_data, Y_data

def load_deepperf_hippacc():
    df = pd.read_csv("HighdimLSE/deepperf_experiment/hipacc_AllNumeric.csv")
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy()
    return X, y

def load_mazda():
    X = np.load("RealworldData/Mazda_CdMOBP/X.npy")
    y = np.load("RealworldData/Mazda_CdMOBP/g07_SUV_SIDE.npy")
    return X, y

def load_mopta08():
    X = np.load("RealworldData/MOPTA08/X.npy")
    y = np.load("RealworldData/MOPTA08/Y.npy")
    return X, y

def load_svm388():
    X = np.load("RealworldData/SVM388/X.npy")
    y = np.load("RealworldData/SVM388/Y.npy")
    return X, y

def prepare_real_world(data_name):
    if data_name=="AA33D":
        X, y = load_deepperf_hippacc()
        fit_and_save_model(X, y, "RealworldData/AlgorithmicAssurance/ground_truth.pkl")
        save_X(X, "data/AA33D/Xtest.pt")
    elif data_name=="Protein20D":
        X, y = data_generate_Protein("HighdimLSE/Protein_data/Protein_data.csv")
        fit_and_save_model(X, y.ravel(), "RealworldData/Protein/ground_truth.pkl")
        save_X(X, "data/Protein20D/Xtest.pt")
    elif data_name=="Mazda74D":
        X, y = load_mazda()
        fit_and_save_model(X, y.ravel(), "RealworldData/Mazda_CdMOBP/ground_truth.pkl")
        save_X(X, "data/Mazda74D/Xtest.pt")
    elif data_name=="Vehicle124":
        X, y = load_mopta08()
        fit_and_save_model(X, y.ravel(), "RealworldData/MOPTA08/ground_truth.pkl")
        save_X(X, "data/Vehicle124/Xtest.pt")

class DataLoader:
    def __init__(
            self, 
            func_params,
            grid=True
        ):
        self.X_range = {i: [func_params["X_min"][i], func_params["X_max"][i]] \
                        for i in range(func_params["n_dims"])}
        self.d = func_params["n_dims"]
        self.test_grid_size = func_params["grid_size"]
        Xtest_path = f"data/{func_params["name"]}/Xtest.pt"
        ytest_path = f"data/{func_params["name"]}/ytest.pt"
        if os.path.exists(Xtest_path) and os.path.exists(ytest_path):
            self.Xtest = torch.load(Xtest_path)
            self.ytest = torch.load(ytest_path)
            if func_params["type"]=="synthetic":
                self.raw_obj_func = func_params["func"]
            else:
                self.raw_obj_func = load_model(func_params["model_path"])
        else:
            os.makedirs(f"data/{func_params["name"]}", exist_ok=True)
            if func_params["type"]=="synthetic":
                self.raw_obj_func = func_params["func"]
                if grid:
                    self.Xtest = self.gen_grid_X(func_params["grid_size"])
                else:
                    # self.Xtest = self.gen_X(min(10000*self.d, 2000000))
                    self.Xtest = self.gen_X(100000)
            else:
                prepare_real_world(func_params["name"])
                self.raw_obj_func = load_model(func_params["model_path"])
                raw_Xtest = torch.load(Xtest_path)
                self.Xtest = norm(raw_Xtest, self.X_range)
            self.ytest = self.raw_obj_func(denorm(self.Xtest, self.X_range))
            torch.save(self.Xtest, Xtest_path)
            torch.save(self.ytest, ytest_path)
        self.y_mean = self.ytest.mean().item()
        self.y_std = self.ytest.std().item()
        self.h_raw = func_params["h"]
        self.ytest = (self.ytest - self.h_raw)/self.y_std
        self.h = 0
        self.data_name = func_params["name"]
        self.batch_size = func_params["batch_size"]
        self.budget = func_params["budget"]

    def standardize(self, Y):
        return (Y - self.h_raw)/self.y_std

    def obj_func(self, X):
        Y = self.raw_obj_func(denorm(X, self.X_range)).unsqueeze(-1).double()
        Y = self.standardize(Y)
        return Y
    
    def noisy_obj_func(self, X):
        Y = self.obj_func(X)
        Y += torch.normal(mean=0, std=1e-4, size=Y.shape)
        return Y
    
    def gen_grid_X(self, grid_size):
        X_list = [list(np.linspace(0, 1, grid_size)) for i in range(self.d)]
        X_grid = torch.from_numpy(np.asarray([item for item in product(*X_list)])).double()
        return X_grid

    def gen_grid_XY(self, grid_size):
        X_grid = self.gen_grid_X(grid_size)
        Y_grid = self.obj_func(X_grid)
        return X_grid, Y_grid

    def gen_X(self, N):
        sobol = SobolEngine(dimension=self.d, scramble=True)
        X_init = sobol.draw(n=N).double()
        return X_init

    def gen_observations(self, N):
        X = self.gen_X(N)
        Y = self.noisy_obj_func(X).double()
        return X, Y

    def gen_observations_raw(self, N):
        X = self.gen_X(N)
        Y = self.raw_obj_func(denorm(X, self.X_range))
        return X, Y