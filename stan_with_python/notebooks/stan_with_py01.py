# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Stanと ~R~ Pythonでベイズ統計モデリング
# ## Stanの基本的な文法
#
#
# data{<br>データの宣言<br>}<br>parameters{<br>サンプリングしたいパラメータ$\theta$<br>}
# <br><br>model{<br>尤度$p(Y|\theta)$<br>事前分布$p(\theta)$<br>}
#
# ### memo
#
# - stanでは、値が決まってなく、確率変数とみなせるものは全てparametersにいれる。
# - 渡辺ベイズによると、データが確率変数の観測値としてみなせるために、事後分布なんかも確率変数と言うこと。
# - 確率モデル、事前分布も自分で定義するもので事後分布も定義してるとみなす。
#
# - 著者によるとモデリングのコツは、
#      1. 最初にモデル部分を記述する
#      2. それから、dataにデータの変数を記述、残りをparametersに記述
#
# の流れで無理に初めから埋めていかないことがコツらしい。

# ##  トイデータで試してみる
#
# $$\begin{eqnarray}Y &\sim& Nonmal(\mu, 1) \\ \mu &\sim& Normal(0, 100)\end{eqnarray}$$
#
# のモデルで、StanでMCMC(Nuts)を使ってパラメータの事後分布を実現させる。
#
# Stanの開発者は、Stringでの記述よりは.Stanファイルに記述することを勧めてる
#
# ここではファイルを分けると行き来がめんどくさいのでStringでモデルを記述する。
#

import pystan
import numpy as np

model = """

data{
    int<lower=0> N; // データの数
    real Y[N];
}

parameters{
    real mu;
}

model{
    for (n in 1:N) {
        Y[n] ~ normal(mu, 1); // normal(mean, std)に注意
    }
    
    mu ~ normal(0, 100);
}

"""

# complie モデルのコンパイル。時間がかかる。
sm = pystan.StanModel(model_code=model)

data = {"N":8, "Y":[15, 10, 16, 11, 9, 11, 10, 18]}
fit = sm.sampling(
        data=data, iter=1000, chains=4
)

fit.plot()

np.mean(data["Y"])

# ## arvizで予測分布まで

J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
schools = np.array(['Choate', 'Deerfield', 'Phillips Andover', 'Phillips Exeter',
                    'Hotchkiss', 'Lawrenceville', "St. Paul's", 'Mt. Hermon'])


# +
schools_model = """
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta[J];
}

model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
generated quantities {
    vector[J] log_lik;
    vector[J] y_hat;
    for (j in 1:J) {
        log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
        y_hat[j] = normal_rng(theta[j], sigma[j]);
    }
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}


sm = pystan.StanModel(model_code=schools_model)
# -

fit = sm.sampling(data=schools_dat, iter=1000, chains=4)

# - パラメータの事後分布の可視化にはpystanではarvizを使う。
#
# 今回MCMCサンプリングしたあとの結果をfitに格納しているので```fit.plot()```もできるがwarningがでて
#
# ```arviz```を使うよう勧められる。
#
# インストールはanacondaを使っていれば```conda install -c conda-forge arviz```
#
# 使い方詳しくは[公式](https://arviz-devs.github.io/arviz/notebooks/Introduction.html)

import arviz as az

az.style.use("arviz-darkgrid")
az.plot_posterior(fit);

print(fit)

az.plot_trace(fit);

az.plot_density(fit);

# 今まではパラメータの事後分布$p(w|X) \propto p(X|w)p(w)$をMCMCサンプリングで近似的に求めたので、
#
# そのパラメータの事後分布で確率モデルを平均した予測分布$p(x^*|X)=\int p(x*|w)p(w|X)$を求める。
#
# ベイズ推定による統計モデリングでは、この予測分布とサンプルを発生している真の確率分布との誤差（汎化誤差）を小さくすることを目指す。
#
# 詳しくは、[ベイズ統計の理論と方法](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/bayes-theory-method.html)や[著者HPの講義資料やQA](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/index-j.html)を参照。
#
# - [arviz.from_pystan document](https://arviz-devs.github.io/arviz/generated/arviz.from_pystan.html)
#
#     - ```cords``` : インデックスとして使われてる値

data = az.from_pystan(
    posterior=fit,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood={'y': 'log_lik'},
    coords={'school': schools},
    dims={
        'theta': ['school'],
        'y': ['school'],
        'log_lik': ['school'],
        'y_hat': ['school'],
        'theta_tilde': ['school']
    }
)
data

data.posterior_predictive

az.plot_pair(data.posterior_predictive, coords={'school': ['Choate', 'Deerfield', 'Phillips Andover']}, divergences=True);

data = az.convert_to_inference_data(fit)
data

data.posterior

data.sample_stats

inference_dist


