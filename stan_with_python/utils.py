import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pystan

from pathlib import Path
from typing import Union, Optional


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


def plot_inference_data(inference_x : np.ndarray, 
                        inference_y : np.ndarray, 
                        save: bool =False, 
                        out_dir: Union[None, str] =None, 
                        png_name: Union[None, str] =None ):
    """
    ベイズ95%信頼区間のプロット。
    mcmcサンプルがこの区間に入る確率が95%.モデルが十分真の分布に対して良いモデルとする。
    """
    mean  = np.mean(inference_y, axis=0)
    quantile_5 = np.quantile(inference_y, 0.05, axis=0)
    upper: np.ndarray = mean + quantile_5
    lower: np.ndarray = mean - quantile_5
    # drawing png
    plt.plot(inference_x, mean, label="mean")
    plt.fill_between(inference_x, upper, lower, alpha=0.4)
    plt.title("inference data and prediction")
    # save png 
    if save:
        if out_dir is None:
            Path.mkdir(os.getcwd(), "save_stanpng")
            out_dir: str = str(Path(os.getcwd(), "save_stanpng"))
        plt.savefig(Path(out_dir, "inference_data_"+png_name))
        plt.close()
    plt.show()
