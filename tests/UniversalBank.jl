#=
10 December 2020

Bank loan classification from
https://www.kaggle.com/sriharipramod/bank-loan-classification/

See notebook at
https://www.kaggle.com/sriharipramod/bank-loan-classification-boosting-technique

=#

using CSV
using Plots
using DataFrames
using Printf
using Statistics

## -------------- load data  -------------- ##
path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests/"
file_name = "tests/UniversalBank.csv"
target = "Personal Loan"
convert_to_one_hot = false

data = CSV.read(file_name, DataFrame)
display(first(data, 5))

## -------------- process data  -------------- ##
# Drop columns which are not significant
data = select!(data, Not(:ID))
data = select!(data, Not(Symbol("ZIP Code")))

X = select(data, Not(target))
Y = select(data, target)
target_vals = Int.(unique(Y)[:, 1])

# add 1 to classes for easy indexing
Y .+= 1
data = select!(data, Not(target))
insertcols!(data, Symbol(target) => Y[:, 1])

if convert_to_one_hot # Convert Categorical Columns to Dummies
    cat_cols = [:Family, :Education] ##,"Personal Loan","Securities Account","CD Account","Online","CreditCard"]
    for col_name in cat_cols
        col = categorical(select(data, col_name)[:, 1])
        # drop this column
        global data = select!(data, Not(col_name))
        # add back one hot encode variables
        codes = [Int(val[1].level) for val in eachrow(col)]
        for level in levels(col)
            insertcols!(data, Symbol(col_name, level) => Int.(codes .== level))
        end
    end
end
println("\nafter processing:")
display(first(data, 5))

# save data
#CSV.write(file_name[1:end-4]*"_onehot.csv", data)
#CSV.write(file_name[1:end-4]*"_cleaned.csv", data)

## -------------- inspect data  -------------- ##
n_samples, n_features = size(data)
n_features -= 1 # exclude the target column
mod = n_features % 2 == 0 ? floor(Int, n_features/2) : floor(Int, (n_features+1)/2)
gd = groupby(data, target)

# plot features
canvases = []
label_canvas = 1 # only put labels on this canvas
for col in names(data)
    if col == string(target)
        continue
    end

    x1 = select(gd[1], col)[:, 1]
    x2 = select(gd[2], col)[:, 1]

    alpha = 0.5
    nbins = 15
    discrete_thres = 5

    bar_width = (max(maximum(x1), maximum(x2)) -
                 min(minimum(x1), minimum(x2)) )/ nbins

    p = plot()
    n_unique = length(unique([x1; x2]))
    bins = (n_unique <= discrete_thres) ? n_unique : :sturges #:auto, :sturges or :fd

    histogram!(p, x1, labels=target_vals[1], alpha=alpha, bins=bins)
    histogram!(p, x2, labels=target_vals[2], alpha=alpha, bins=bins)
    xlabel!(col)

    push!(canvases, p)
end
if n_features % 2 != 0
    push!(canvases, plot(axis=nothing, showaxis=false)) # dummy plot
end
ylabel!(canvases[1], "frequency")
ylabel!(canvases[mod+1], "frequency")
p1 = plot(canvases..., layout=(2, mod), legend=false, link = :y, size=(1000, 550))
plot!(canvases[1], legend=true)
plot!(canvases[2], title="features")
display(p1)
savefig(p1, "UniversalBank_features_jl.png")

#plot target class
r = target_vals .+ 1
counts = [count(Y[:,1].==i) for i in r]
n = sum(counts)
angles = vcat(0, cumsum(counts)/n * 2π)
mids = [(angles[i]+angles[i+1])/2 for i in r] # midpoint of each segment

annotations = collect((0.6cos(mids[i]), 0.6sin(mids[i]),
                       @sprintf("%.1f%%\n(%d)", 100counts[i]/n, counts[i]))
                      for i in r) #(x, y, text)
p2 = pie(target_vals, counts, legend=:topleft, title="Target variable: $target",
   annotations=annotations, startangle=90)
display(p2)
savefig("UniversalBank_target_jl.png")
