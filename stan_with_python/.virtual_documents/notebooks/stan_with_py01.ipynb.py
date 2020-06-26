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


J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
schools = np.array(['Choate', 'Deerfield', 'Phillips Andover', 'Phillips Exeter',
                    'Hotchkiss', 'Lawrenceville', "St. Paul's", 'Mt. Hermon'])



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


fit = sm.sampling(data=schools_dat, iter=1000, chains=4)


import arviz as az


az.style.use("arviz-darkgrid")
az.plot_posterior(fit);


print(fit)


az.plot_trace(fit);


az.plot_density(fit);


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



