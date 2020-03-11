data{
    int N;
    int K;
    real X[N];
    real Y[N];
    int<lower=1, upper=K> KID[N];
}

parameters{
    real a0;// a_全体平均
    real b0;// b_全体平均
    real ak[K];// a_会社差
    real bk[K];// b_会社差
    real<lower=0> s_a; //　sigam_a
    real<lower=0> s_b; // sigma_b
    real<lower=0> s_Y; //
}

transformed parameters{
    real a[K];
    real b[K];
    for (k in 1:K){
        a[k] = a0 + ak[k]; 
        b[k] = b0 + bk[k]; 
    }
}

model{
    for (k in 1:K){
        ak[k] ~ normal(0, s_a); // a_会社差について平均０標準偏差s_aの正規分布を仮定
        bk[k] ~ normal(0, s_b); // b_会社差について平均０標準偏差s_bの正規分布を仮定
    }  

    for (i in 1:N)
        Y[i] ~ normal(a[KID[i]] + b[KID[i]] * X[i], s_Y); 
}