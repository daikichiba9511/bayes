import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import pystan
import scipy.stats as ss

from criterion import Criterion
import utils

import pickle


def main():
    base_dir = Path.cwd() /"stan_with_python"
    data = pd.read_csv(base_dir / "input" / "data-ss2.txt")
    standata ={
        "T":np.max(data["X"]),
        "Y":data["Y"]
    }
    
    stan_path = base_dir / "model" / "stanmodel" / "model12-6.stan"
    pkl_path = base_dir / "model" / "model_pkl" / "model12-6.pkl"
    sm12_6 = utils.read_stanmodel(stan_path, pkl_path)
    fit12_6 = sm12_6.sampling(
        data=standata,
        iter=3000,
        seed=496,
        warmup=300,
        chains=4
    )
    plt.figure(figsize=(10,8))
    plt.style.use("ggplot")
    utils.plot_inference_interval(original_x=data["X"],
                            original_y=data["Y"],
                            inference_mean=fit12_6.extract()["y_mean"],
                            pred_x=data["X"])



if __name__ == "__main__":
    main()