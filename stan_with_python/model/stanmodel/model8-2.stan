

data {
    int N;
    int K;
    real X[N];
    real Y[N];
    int<lower=0, upper=K> KID[N];
}

parameters {
    real a[K];
    real b[K];
    real<lower=0> s_Y;
} 

model {
    for (i in 1:N)
        Y[i] ~ normal(a[KID[i]] + b[KID[i]]*X[i], s_Y);
}