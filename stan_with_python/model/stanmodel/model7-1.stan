data{
    int N;
    real Area[N];
    real Y[N];
    real sigma;
}

parameters{
    real b[2];
}

transformed parameters{
    real mu[N];

    for (i in 1:N)
        mu[i] = b[1] + b[2] * Area[i];
}

model{
    for (i in 1:N)
        Y[i] ~ normal(mu[i], sigma);
}

generated quantities{
    real y_pred[N];
    for (i in 1:N)
        y_pred[i] = normal_rng(mu[i], sigma);
}