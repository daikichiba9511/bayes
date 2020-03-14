import os
import pickle
import pystan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


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
    lower , upper = ss.mstats.mquantiles(inference_y, [0.0025, 0.975], axis=0)
    # drawing png
    plt.plot(inference_x, mean, label="mean")
    plt.fill_between(inference_x, upper, lower, alpha=0.4)
    plt.title("inference data and prediction")
    plt.legend()
    # save png 
    if save:
        if out_dir is None:
            Path.mkdir(os.getcwd(), "save_stanpng")
            out_dir: str = str(Path(os.getcwd(), "save_stanpng"))
        plt.savefig(Path(out_dir, "inference_data_"+png_name))
        plt.close()
    plt.show()

def plot_inference_interval(original_x,
                            original_y,
                            inference_mean,
                            pred_x):
    """
    orginal_x : 解析に使ったデータのX
    original_y :　解析に使ったデータのY
    inference_mean : stanで求めた平均
    pred_x :　inference_meanの数
    """
    # draw
    plt.plot(original_x, original_y, marker="o")
    lower80 , upper80 = ss.mstats.mquantiles(inference_mean, [0.1, 0.9], axis=0)
    lower50 , upper50 = ss.mstats.mquantiles(inference_mean, [0.25, 0.75], axis=0)
    plt.plot(pred_x, np.mean(inference_mean, axis=0))
    plt.fill_between(pred_x, lower80, upper80, alpha=0.3)
    plt.fill_between(pred_x, lower50, upper50, alpha=0.4)
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.show()