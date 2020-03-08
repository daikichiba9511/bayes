
data {
    int N;
    int<lower=0, upper=1> A[N];
    real<lower=0, upper=1> Score[N];
    int<lower=0> M[N];

}

parameters {
    real b[3];
}

transformed parameters {
    real lambda[N];
    for (i in 1:N)
        lambda[i] = exp(b[1] + b[2] * A[i] + b[3] * Score[i]);
}

model {
    for (i in 1:N)
        M[i] ~ poisson(lambda[i]);
}

generated quantities { // 近似した事後分布に従って、予測分布を生成してる。
    int m_pred[N];
    for (i in 1:N)
        m_pred[i] = poisson_rng(lambda[i]);
}