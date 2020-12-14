#=
10 December 2020

High level type definitions

=#

using DataFrames
using Statistics
import Base: showerror

abstract type AbstractClassifier end
# this type should have the following methods: predict, score, fit!

struct NotFittedError <: Exception
    var::Symbol
end

Base.showerror(io::IO, e::NotFittedError) = print(io, e.var, " has not been fitted to a dataset. Call fit!($(e.var), X, Y) first")

"""
    predict(classfier::AbstractClassifier, X::DataFrame) => Y

Return a prediction of classes for each row in X
"""
function predict(classfier::AbstractClassifier, X::DataFrame)
    throw("predict not implemented for classifier of type $(typeof(classfier))")
end

"""
    score(classfier::AbstractClassifier, X::DataFrame, Y::DataFrame) => Float64

The proportion of predictions made with X that match Y. Range is [0, 1].
"""
function score(classfier::AbstractClassifier, X::DataFrame, Y::DataFrame)
    y_pred = predict(classfier, X)
    return count(y_pred .== Y[:, 1]) / size(Y, 1)
end

"""
    fit!(classfier::AbstractClassifier, X::DataFrame, Y::DataFrame) => nothing

Fit a classifier to output the classes in Y based on the data in X.
"""
function fit!(classifier::AbstractClassifier, X::DataFrame, Y::DataFrame)
    throw(error("fit! not implemented for classifier of type $(typeof(classifier))"))
end

"""
    perm_feature_importance(classifier::AbstractClassifier,  X::DataFrame, Y::DataFrame; n_repeats=10, random_state=Random.GLOBAL_RNG) => Dict(:means, :stds)

Calculate feature importance based on random permutations of each feature column.
The larger the drop in accuracy from shuffling each column, the higher the feature importance.
"""
function perm_feature_importance(classifier::AbstractClassifier,  X::DataFrame, Y::DataFrame;
                                 n_repeats=10, random_state=Random.GLOBAL_RNG)
        y_pred = predict(classifier, X)
        y = Y[:, 1]
        acc_full = mean(y_pred .== y)
        n_features = size(X, 2)

        feature_importances = zeros(n_repeats, n_features)
        for (j, col) in enumerate(names(X))
            X_rand = copy(X)
            for i in 1:n_repeats
                X_rand[:, j] .= shuffle(random_state, X_rand[:, j])
                y_pred .= predict(classifier, X_rand)
                feature_importances[i, j] = acc_full - mean(y_pred .== y)
            end
        end

        return Dict(
            :means => mean(feature_importances, dims=1)[1, :],
            :stds => std(feature_importances, dims=1)[1, :]
            )
end
