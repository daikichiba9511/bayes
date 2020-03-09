import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pystan

from pathlib import Path


def read_stanmodel(stan_path, pickle_path):
        
    try:
        with open(pickle_path, "rb") as f:
            print("loading...", pickle_path)
            stanmodel = pickle.load(f)

    except FileNotFoundError:
        print("save path to stan file is ", stan_path)
        stanmodel = pystan.StanModel(
            file = str(stan_path),
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(stanmodel, f)
            print("saving finished...")
    return stanmodel


def plot_inference_data(inference_x : pd.DataFrame, 
                        inference_y : pd.DataFrame, 
                        interval_coef=1.96, save=False, out_dir=None, png_name=None):
    """これ正規分布モデルの時だけ。。。
    """
    mean = np.mean(inference_y, axis=0)
    std = np.std(inference_y,  axis=0)
    # print(f"mean: {mean.shape} \n std: {std.shape}")
    upper: np.array = mean + interval_coef * std
    lower: np.array = mean - interval_coef * std
    # drawing png
    plt.plot(inference_x, mean, label="mean")
    plt.fill_between(inference_x, upper, lower, alpha=0.4)
    plt.title("inference data and prediction")
    # save png 
    if save:
        if out_dir is None:
            Path.mkdir(os.getcwd(), "save_stanpng")
            out_dir = str(Path(os.getcwd(), "save_stanpng"))
        plt.savefig(Path(out_dir, "inference_data_"+png_name))
        plt.close()
    plt.show()
