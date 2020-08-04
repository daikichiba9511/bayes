# plot fig. 4.2
# data file to plot scatter is data-salary.txt in input directory
# the file is getted from https://github.com/MatsuuraKentaro/RStanBook

using PyPlot
using DataFrames
using CSV

function read_data()
    CSV.read("./input/data-salary.txt", DataFrame)
end

function plot_scatter_fig4_2()
    df = read_data()
    # confirm dataframes by show macro
    @show df
    @show df[!, 1];
    @show df[!, 2];
    # create fig object of pyplot
    fig = figure()
    scatter(df[!, 1], df[!, 2])
    title("fig4.2")
    xlabel("X"); ylabel("Y");
    grid()
    display(fig)
    savefig("stan_with_julia/Chapter04/fig4-2.png")
    close(fig)
end

plot_scatter_fig4_2()
