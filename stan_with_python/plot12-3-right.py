import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import pystan
import scipy.stats as ss

from criterion import Criterion
import utils

import pickle

def plot_inference_interval(original_x,
                            original_y,
                            inference_mean,
                            pred_x,
                            save=False,
                            out_dir=None,
                            png_name=None):
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
    # save png 
    if save:
        if out_dir is None:
            base_dir = Path.cwd()
            Path.mkdir(base_dir / "save_stanpng")
            out_dir = base_dir / "save_stanpng"
        plt.savefig(out_dir / "inference_data_"+png_name)
        plt.close()
    plt.show()

def main():
    base_dir = Path.cwd() / "stan_with_python"
    # data
    data = pd.read_csv(base_dir/"input"/"data-ss1.txt")
    standata = {
        "T":data.shape[0],
        "Y":data["Y"],
        "T_pred":3
    }

    # stanmodel
    stan_path = base_dir /"model"/ "stanmodel" / "model12-3.stan"
    pkl_path = base_dir /"model"/ "model_pkl" / "model12-3.pkl"
    sm12_3 = utils.read_stanmodel(stan_path, pickle_path=pkl_path)
    fit12_3 = sm12_3.sampling(
        data=standata,
        iter=3000,
        seed=496,
        warmup=300,
        chains=4
    )

    ms12_3 = fit12_3.extract()
    x = [i for i in range(1, 25)]
    plot_inference_interval(original_x=data["X"],
                            original_y=data["Y"],
                            inference_mean=ms12_3["mu_all"],
                            pred_x=x
    )

if __name__ == "__main__":
    main()
