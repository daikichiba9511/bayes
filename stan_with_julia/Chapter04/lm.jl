using DataFrames
using GLM
using CSV
using PyPlot

function read_data()
    CSV.read("./input/data-salary.txt", DataFrame)
end


function lm_predict()
    println("====== lm fit and predict start =======")
    df = read_data()
    res_lm = lm(@formula(Y ~ X), df)

    # create new data to be predicted
    df_new = DataFrame(X=23:60)
    @show df_new

    conf_95 = predict(res_lm, df_new, interval=:confidence, level=0.95)
    pred_95 = predict(res_lm, df_new, interval=:prediction, level=0.95)
end


function plot_fig4_3()
    df = read_data()
    res_lm = lm(@formula(Y ~ X), df)

    # create new data to be predicted
    X_new = 23:60
    df_new = DataFrame(X=X_new)
    # @show df_new


    # 95%prediction interval
    conf_95 = predict(res_lm, df_new, interval=:confidence, level=0.95)
    pred_95 = predict(res_lm, df_new, interval=:prediction, level=0.95)
    
    # 50％prediction interval
    conf_50 = predict(res_lm, df_new, interval=:confidence, level=0.50)
    pred_50 = predict(res_lm, df_new, interval=:prediction, level=0.50)
    # pred_50 = predict(res_lm, df_new, interval=:prediction, level=0.50)

    function plot_fig4_3_right()
        # plot 95% prediction interval and 50% one
        fig = figure()
        scatter(df.X, df.Y, color=:green)
        plot(df_new.X, pred_95.prediction)
        fill_between(df_new.X, pred_95.upper, pred_95.lower, alpha=0.3)
        fill_between(df_new.X, pred_50.upper, pred_50.lower, alpha=0.6)
        xlabel("X_new") ; ylabel("prediction")
        title("plot prediction with 95% prediction interval and 50% prediction interval")
        grid()
        display(fig)
        savefig("stan_with_julia/Chapter04/fig4-plot_fig4_3_right.png")
        close(fig)
    end

    function plot_fig4_3_left()
        # plot 95% confidence interval and 50% one
        fig = figure()
        scatter(df.X, df.Y, color=:green)
        plot(X_new, conf_95.prediction)
        fill_between(df_new.X, conf_95.upper, conf_95.lower, alpha=0.3)
        fill_between(df_new.X, conf_50.upper, conf_50.lower, alpha=0.6)
        xlabel("X_new") ; ylabel("prediction")
        title("plot prediction with 95% confidence interval and 50% confidence interval")
        grid()
        display(fig)
        savefig("stan_with_julia/Chapter04/fig4-plot_fig4_3_left.png")
        close(fig)
        close(fig)
    end

    plot_fig4_3_right()
    plot_fig4_3_left()
end

# lm_predict()
plot_fig4_3()
close(fig)
