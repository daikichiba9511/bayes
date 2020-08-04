using DataFrames
using CSV
using PyPlot

using StanSample
using ArviZ


function read_data()
    CSV.read("./input/data-salary.txt", DataFrame)
end

function run_stan_defined_by_string()
    df = read_data()
    observed_sample = Dict(
        "N" => size(df, 1),
        "X" => df["X"],
        "Y" => df["Y"],
    )

    # load model in stan file
    f = "
    data {
        int N;
        real X[N];
        real Y[N];
    }
    
    # prior disttribution is Unioform Distribution
    parameters {
        real a;
        real b;
        real<lower=0> sigma;
    }
    
    model {
        for (n in 1:N) {
            Y[n] ~ normal(a + b*X[n] , sigma);
        }
    }
    "
    sm = SampleModel("model4-5", f)
    sm |> display
    fitted = stan_sample(sm, data=observed_sample)
end

function run_stan_defined_by_stanfile()
    df = read_data()
    observed_sample = Dict(
        "N" => size(df, 1),
        "X" => df["X"],
        "Y" => df["Y"],
    )

    # load model in stan file as String
    f = open("./Chapter04/model/model4-5.stan", "r")
    model = read(f, String)
    close(f)

    # create Stan instance
    sm = SampleModel(
        "model4-5", model;
        method=StanSample.Sample(num_samples=10000),
    )
    sm |> display

    # sampling : generate mcmc samples from posterior dist.
    # posterior dist. is conditional distribution that the data is obtained in the dist. by chance.
    rc = stan_sample(sm; data=observed_sample, seed=42)
    if success(rc)
        # mcmc samples
        samples = read_samples(sm)

        fitted = read_summary(sm)
        fitted |> display
        
        # save dataframe as CSV file
        CSV.write("Chapter04/output/mcmc_summary.csv", fitted)
    end
end

function run_stan_and_plot()
    df = read_data()
    observed_sample = Dict(
        "N" => size(df, 1),
        "X" => df["X"],
        "Y" => df["Y"],
    )

    # load model in stan file as String
    f = open("./Chapter04/model/model4-5.stan", "r")
    model = read(f, String)
    close(f)

    # create Stan instance
    sm = SampleModel(
        "model4-5", model;
        method=StanSample.Sample(num_samples=10000),
    )
    sm |> display

    # sampling : generate mcmc samples from posterior dist.
    # posterior dist. is conditional distribution that the data is obtained in the dist. by chance.
    rc = stan_sample(sm; data=observed_sample, seed=42)
    if success(rc)
        # mcmc samples
        samples = read_samples(sm)

        fitted = read_summary(sm)
        fitted |> display
        
        # save dataframe as CSV file
        CSV.write("Chapter04/output/mcmc_summary.csv", fitted)

        println(typeof(rc), typeof(sm))

        plot_posterior(samples)
        # plot_trace(samples)
        display(gcf())
        close(gcf())
        #= 
        # if 'generated_quantities' is defined in stan file , then following code is used. So, this is memo for me.
        # but, haven't yet confirmed that we can operate following one.
        # generate samples which is from prediction distribution .

        StanSample.stan_generate_quantities(sm, 1); # indicated id

        # read_generated_quantities() is that read generated_quantities output files created by StanSample.jl
        # require arguments is 'model' which type is 'StanSample'
        (y_preds, parameters) = read_generated_quantities(sm, 2);
        y_preds |> display
        parameters |> display
        =#
    end

end

run_stan_and_plot()