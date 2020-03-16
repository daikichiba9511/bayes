import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as ss

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import chart_studio.plotly as py
import plotly.graph_objs as go

plt.style.use("ggplot")

def log_likelihood(x, a, b ):
    def normal_pdf(x):
        pdf =  ( 1 / np.sqrt(2 * np.pi) ) * np.exp( - (1 / 2) * x ** 2)
        return pdf
    def mixture_normal(x, a, b):
        return (1 - a) * normal_pdf(x) + a * normal_pdf(x-b)
    return np.sum(np.log(mixture_normal(x, a, b)), axis=0)
def q_x(a, b, sample_size):
    return (1 - a ) * np.random.normal(size=sample_size) + a * np.random.normal(loc=b, size=sample_size)

def main():
    # 真の分布からのデータ
    # W = (a_0, b_0)
    W = [(0.5, 3.0), (0,5, 1.0), (0.5, 0.5)]
    # data1 = q_x(a=W[0][0], b=W[0][1], sample_size=200)
    # data2 = q_x(a=W[1][0], b=W[1][1], sample_size=100)
    data3 = q_x(a=W[2][0], b=W[2][1], sample_size=100)
    data = data3
    a_i = np.linspace(0.0, 1.0, num=data.shape[0])
    b_i = np.linspace(-5.0, 5.0, num=data.shape[0])
    
    # data1の時の混合ガウス分布の確率密度関数
    a, b = np.meshgrid(a_i, b_i)
    posteriors = [[0]*data.shape[0] for _ in range(data.shape[0])]
    for c , i in enumerate(a_i):
        for r, j in enumerate(b_i):
            posterior_ = np.exp(log_likelihood(data, a=i, b=j)) * 1e+100
            posteriors[c][r] = posterior_

    post = np.array(posteriors)

    # plotly
    plotly_data = [
        go.Surface(
            x=b,
            y=a,
            z=post
        )
    ]
    layout = go.Layout(
        title='posterior',
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=plotly_data, layout=layout)
    fig.show()

if __name__ == "__main__":
    main()