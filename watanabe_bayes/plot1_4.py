# ベイズ統計の理論と方法　p18~ 事後分布の例で事後分布のプロット。
# ただし、事前分布に一様分布を用いてるので形状は尤度関数と同じなので尤度関数の形状をプロットしてる
# data1, data2, data3をdata

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objs as go

def normal_pdf(x):
    pdf =  ( 1 / np.sqrt(2 * np.pi) ) * np.exp( - (1 / 2) * x ** 2)
    return pdf

def mixture_normal(x, a, b):
    return (1 - a) * normal_pdf(x) + a * normal_pdf(x-b)

def log_likelihood(x, a, b ): 
    return np.sum(np.log(mixture_normal(x, a, b)), axis=0)

def q_x(a, b, sample_size):
    return (1 - a ) * np.random.normal(size=sample_size) + a * np.random.normal(loc=b, size=sample_size)

def plot(data):
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


def main():
    # 真の分布からのデータ
    # W = (a_0, b_0)
    W = [(0.5, 3.0), (0,5, 1.0), (0.5, 0.5)]
    # data1 = q_x(a=W[0][0], b=W[0][1], sample_size=200)
    data2 = q_x(a=W[1][0], b=W[1][1], sample_size=100)
    # data3 = q_x(a=W[2][0], b=W[2][1], sample_size=100)
    data = data2
    plot(data)
    

if __name__ == "__main__":
    main()