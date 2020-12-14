
#=
9 December 2020

Iris flower dataset

=#

using CSV
using Plots
using DataFrames
using Printf
using Statistics

## -------------- load data  -------------- ##
path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests"
file_name = "tests/Iris.csv"
target = "Species"

data = CSV.read(file_name, DataFrame)

## -------------- process data  -------------- ##
# drop Id column
data = select!(data, Not(:Id))

#change classes to categorical data. replcae with code values
cv = categorical(data.Species)
species_names = levels(cv[:, 1])
codes = [Int(val[1].level) for val in eachrow(cv)]

data = select!(data, names(data).!=target)
insertcols!(data, :Species => codes)

# save data
#CSV.write(file_name[1:end-4]*"_cleaned.csv", data)

println("\nafter processing:")
display(first(data, 5))

## -------------- inspect data  -------------- ##
gd = groupby(data, target)

X = select(data, Not(target))
Y = select(data, target)

# plot features
canvases = []
label_canvas = 1 # only put labels on this canvas
for col in names(data)
    if col == string(target)
        continue
    end

    x1 = select(gd[1], col)[:, 1]
    x2 = select(gd[2], col)[:, 1]
    x3 = select(gd[3], col)[:, 1]

    alpha = 0.5
    nbins = 15
    discrete_thres = 5

    bar_width = (max(maximum(x1), maximum(x2), maximum(x3)) -
                 min(minimum(x1), minimum(x2), minimum(x3)) )/ nbins

    # Nice ideas for getting good plots at
    # https://nextjournal.com/leandromartinez98/tips-to-create-beautiful-publication-quality-plots-in-julia

    p = plot()
    histogram!(p, x1, labels=species_names[1], alpha=alpha, bar_width=bar_width)
    #nbins = round(Int64,( maximum(x2) - minimum(x2) ) / bin_width)
    histogram!(p, x2, labels=species_names[2], alpha=alpha, bar_width=bar_width)
    #nbins = round(Int64,( maximum(x3) - minimum(x3) ) / bin_width)
    histogram!(p, x3, labels=species_names[3], alpha=alpha, bar_width=bar_width)
    xlabel!(col)

    push!(canvases, p)
end
ylabel!(canvases[1], "frequency")
title!(canvases[2], "Distribution of features in data set")
p1 = plot(canvases..., layout=(1, length(canvases)), legend=false, size=(1000, 550), link=:y)
plot!(canvases[1], legend=true)
savefig(p1, "Iris_features_jl.png")
display(p1)

#plot target class
counts = [count(Y[:,1].==i) for i in 1:3]
n = sum(counts)
angles = vcat(0, cumsum(counts)/n * 2π)
mids = [(angles[i]+angles[i+1])/2 for i in 1:3] # midpoint of each segment

annotations = collect((0.5cos(mids[i]), 0.5sin(mids[i]),
                        @sprintf("%.1f%%\n(%d)", 100counts[i]/n, counts[i]))
                       for i in 1:3) #(x, y, text)
p2 = pie(species_names, counts, legend=:topleft, title="Target variable: Species",
    annotations=annotations)
display(p2)
savefig("Iris_target_jl.png")
