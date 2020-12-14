#=
10  December 2020

Random Forests from scratch. Redo of Python code

Sources
- https://course18.fast.ai/lessonsml1/lesson5.html
- https://github.com/bensadeghi/DecisionTree.jl

=#


using Random
using CSV, DataFrames, Printf

include("DecisionTree.jl")
#include("Utilities.jl")  # already in DecisionTree.jl
#include("Classifier.jl") # already in DecisionTree.jl

## --------------------- Random Forest Classifier  --------------------- ##
"""
    RandomForestClassifier{T}([n_trees=100], [max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG], [bootstrap=true], [oob_score=false])
    RandomForestClassifier([n_trees=100], [max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG], [bootstrap=true], [oob_score=false]) -> T = Float64

A random forest classifier.

Available methods are:
`fit!`, `predict`, `predict_prob`, `score`,
`feature_importance_impurity`, `perm_feature_importance`
"""
mutable struct RandomForestClassifier{T}  <: AbstractClassifier
    T::DataType #for the type of values in the DecisionTree.
    #internal variables
    n_features::Union{Int, Nothing}
    n_classes::Union{Int, Nothing}
    features::Vector{String}
    trees::Vector{DecisionTreeClassifier}
    feature_importances::Union{Vector{Float64}, Nothing}

    # external parameters
    n_trees::Int
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing} # sets n_features_split
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}
    bootstrap::Bool
    oob_score::Bool
    oob_score_::Union{Float64, Nothing}

    RandomForestClassifier{T}(;
            n_trees=100,
            max_depth=nothing,
            max_features=nothing,
            min_samples_leaf=1,
            random_state=Random.GLOBAL_RNG,
            bootstrap=true,
            oob_score=false
        ) where T = new(T,
            nothing, nothing, [], [], nothing, n_trees,
            max_depth, max_features, min_samples_leaf, check_random_state(random_state), bootstrap, oob_score, nothing
            )
end
RandomForestClassifier(;
        n_trees=100,
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=Random.GLOBAL_RNG,
        bootstrap=true,
        oob_score=false
    ) = RandomForestClassifier{Float64}(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        bootstrap=bootstrap,
        oob_score=oob_score
    )


function Base.show(io::IO, forest::RandomForestClassifier)
    rng = forest.random_state
    if hasproperty(rng, :seed)
        str_rng = string(typeof(rng), "($(forest.random_state.seed),...)")
    else
        str_rng = string(typeof(rng))
    end
    str_out = string(
        typeof(forest), "(",
        "n_trees=$(forest.n_trees)",
        ", max_depth=$(forest.max_depth)",
        ", max_features=$(forest.max_features)",
        ", min_samples_leaf=$(forest.min_samples_leaf)",
        ", random_state="*str_rng,
        ", bootstrap=$(forest.bootstrap)",
        ", oob_score=$(forest.oob_score)",
        ")"
    )
    println(io, str_out)
end

## --------------------- fitting --------------------- ##
function fit!(forest::RandomForestClassifier, X::DataFrame, Y::DataFrame)
    @assert size(Y, 2) == 1 "Output Y must be an m x 1 DataFrame"

    # set internal variables
    forest.n_features = size(X, 2)
    forest.n_classes = size(unique(Y), 1)
    forest.features = names(X)

    # create decision trees
    rng_states = typeof(forest.random_state)[]  # save the random states to regenerate the random indices for the oob_score
    for i in 1:forest.n_trees
        push!(rng_states, copy(forest.random_state))
        push!(forest.trees, create_tree(forest, X, Y))
    end

    # set attributes
    forest.feature_importances = feature_importance_impurity(forest)
    if forest.oob_score
        if !forest.bootstrap
            println("Warning: out-of-bag score will not be calculated because bootstrap=false")
        else
            forest.oob_score_ = calculate_oob_score(forest, X, Y, rng_states)
        end
    end

    return
end

function create_tree(forest::RandomForestClassifier, X::DataFrame, Y::DataFrame)
    n_samples = nrow(X)

    if forest.bootstrap # sample with replacement
        idxs = [rand(forest.random_state, 1:n_samples) for i in 1:n_samples]
        X_ = X[idxs, :]
        Y_ = Y[idxs, :]
    else
        X_ = copy(X)
        Y_ = copy(Y)
    end

    new_tree = DecisionTreeClassifier{forest.T}(
            max_depth = forest.max_depth,
            max_features = forest.max_features,
            min_samples_leaf = forest.min_samples_leaf,
            random_state = forest.random_state
    )
    fit!(new_tree, X_, Y_)

    return new_tree
end

function calculate_oob_score(
    forest::RandomForestClassifier, X::DataFrame, Y::DataFrame,
    rng_states::Vector{T}) where T <: AbstractRNG
    n_samples = nrow(X)
    oob_prob  = zeros(n_samples, forest.n_classes)
    oob_count = zeros( n_samples)
    for (i, rng) in enumerate(rng_states)
        idxs = Set([rand(forest.random_state, 1:n_samples) for i in 1:n_samples])
        # note: expected proportion of out-of-bag is 1-exp(-1) = 0.632...
        # so length(row_oob)/n_samples ≈ 0.63
        row_oob =  filter(idx -> !(idx in idxs), 1:n_samples)
        oob_prob[row_oob, :] .+= predict_prob(forest.trees[i], X[row_oob, :])
        oob_count[row_oob] .+= 1.0
    end
    # remove missing values
    valid = oob_count .> 0.0
    oob_prob = oob_prob[valid, :]
    oob_count = oob_count[valid]
    y_test = Y[valid, 1]
    # predict out-of-bag score
    y_pred = mapslices(argmax, oob_prob./oob_count, dims=2)[:, 1]
    return mean(y_pred .==  y_test)
end



## --------------------- prediction --------------------- ##
function predict_prob(forest::RandomForestClassifier, X::DataFrame)
    if length(forest.trees) == 0
        throw(NotFittedError(:forest))
    end
    probs = zeros(nrow(X), forest.n_classes)
    for tree in forest.trees
        probs .+= predict_prob(tree, X)
    end
    return probs
end

function predict(forest::RandomForestClassifier, X::DataFrame)
    probs = predict_prob(forest, X)
    return mapslices(argmax, probs, dims=2)[:, 1]
end

## --------------------- description --------------------- ##
"""
    nleaves(forest::RandomForestClassifier) => Vector{Int}

The number of leaves in each DecisionTreeClassifier in the forest.
"""
nleaves(forest::RandomForestClassifier) = [nleaves(tree.binarytree) for tree in forest.trees]

"""
    feature_importance_impurity(forest::RandomForestClassifier) => Vector{Float64}

Calculate feature importance based on impurity.
This is the mean of all the impurity feature importances for each DecisionTreeClassifier in the forest for each feature.
"""
function feature_importance_impurity(forest::RandomForestClassifier)
    if length(forest.trees) == 0
        throw(NotFittedError(:forest))
    end
    feature_importances = zeros(forest.n_trees, forest.n_features)
    for (i, tree) in enumerate(forest.trees)
        feature_importances[i, :] = tree.feature_importances
    end
    return mean(feature_importances, dims=1)[1, :]
end
