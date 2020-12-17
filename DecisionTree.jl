#=
8 December 2020

Random Forests from scratch. Redo of Python code

Decision tree
- Tree is represent with 2 parallel arrays. This is more compact and requires much recusion than a linked list.
    - left_child_id = tree_.children_left[parent_id]
    - right_child_id = tree_.children_left[parent_id]
    - if id = -1, this node does not exist
- Works for multi-class problems

see https://github.com/bensadeghi/DecisionTree.jl

=#

using Random
using CSV, DataFrames
using Printf
import Base: size

gini_score(counts) = 1.0 - sum(counts .* counts)/(sum(counts) ^2)

## --------------------- Binary Tree --------------------- ##
"""
    BinaryTree()

A binary tree implemented as 2 parallel arrays.

Available methods are: `add_node!`, `set_left_child`, `set_right_child`, `get_children`, `is_leaf`, `size`, `nleaves`, `find_depths`, `get_max_depth`
"""
mutable struct BinaryTree
    children_left::Vector{Int}
    children_right::Vector{Int}
    BinaryTree() = new([], [])
end

function add_node!(tree::BinaryTree)
    push!(tree.children_left, -1)
    push!(tree.children_right, -1)
    return
end

function set_left_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_left[node_id] = child_id
    return
end

function set_right_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_right[node_id] = child_id
    return
end

function get_children(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id], tree.children_right[node_id]
end

function is_leaf(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id] == tree.children_right[node_id] == -1
end

nleaves(tree::BinaryTree) = count(tree.children_left .== -1)
size(tree::BinaryTree) = length(tree.children_left)

function find_depths(tree::BinaryTree)
    depths = zeros(Int, size(tree))
    depths[1] = -1
    stack = [(1, 1)] # (parent, node_id)
    while !isempty(stack)
        parent, node_id = pop!(stack)
        if node_id == -1
            continue
        end
        depths[node_id] = depths[parent] + 1
        left, right = get_children(tree, node_id)
        push!(stack, (node_id, left))
        push!(stack, (node_id, right))
    end
    return depths
end

"""
    get_max_depth(tree::BinaryTree; node_id=0) => Int

Calculate the maximum depth of the tree
"""
function get_max_depth(tree::BinaryTree; node_id=1)
    if is_leaf(tree, node_id)
        return 0
    end
    left, right = get_children(tree, node_id)
    return max(get_max_depth(tree, node_id=left), get_max_depth(tree, node_id=right)) + 1
end



## --------------------- Decision Tree Classifier --------------------- ##
"""
    DecisionTreeClassifier{T}([max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG])
    DecisionTreeClassifier([max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG]) -> T=Float64

A random forest classifier.

Available methods are:
`fit!`, `predict`, `predict_prob`, `predict_row`, `predict_batch`, `score`,
`feature_importance_impurity`, `perm_feature_importance`
`print_tree`, `node_to_string`
"""
mutable struct DecisionTreeClassifier{T} <: AbstractClassifier
    #internal variables
    num_nodes::Int
    binarytree::BinaryTree
    n_samples::Vector{Int} # total samples per each node
    values::Vector{Vector{Float64}} # samples per class per each node. Float64 to speed up calculations
    impurities::Vector{Float64}
    split_features::Vector{Union{Int, Nothing}}
    split_values::Vector{Union{T, Nothing}} #Note: T is the same for all values
    n_features::Union{Int, Nothing}
    n_classes::Union{Int, Nothing}
    features::Vector{String}
    feature_importances::Union{Vector{Float64}, Nothing}

    # external parameters
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing} # sets n_features_split
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}

    DecisionTreeClassifier{T}(;
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=Random.GLOBAL_RNG
        ) where T = new(
            0, BinaryTree(), [], [], [], [], [], nothing, nothing, [], nothing,
            max_depth, max_features, min_samples_leaf, check_random_state(random_state)
            )
end
DecisionTreeClassifier(;
    max_depth=nothing,
    max_features=nothing,
    min_samples_leaf=1,
    random_state=Random.GLOBAL_RNG
) = DecisionTreeClassifier{Float64}(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=check_random_state(random_state)
    )


function Base.show(io::IO, tree::DecisionTreeClassifier)
    rng = tree.random_state
    if hasproperty(rng, :seed)
        str_rng = string(typeof(rng), "($(tree.random_state.seed),...)")
    else
        str_rng = string(typeof(rng))
    end
    str_out = string(
        typeof(tree), "(",
        "num_nodes=$(tree.num_nodes)",
        ", max_depth=$(tree.max_depth)",
        ", max_features=$(tree.max_features)",
        ", min_samples_leaf=$(tree.min_samples_leaf)",
        ", random_state="*str_rng,
        ")"
    )
    println(io, str_out)
end

function node_to_string(tree::DecisionTreeClassifier, node_id::Int)
    n_samples = tree.n_samples[node_id]
    value = tree.values[node_id]
    impurity = tree.impurities[node_id]
    s = @sprintf("n_samples: %d; value: %s; impurity: %.4f",
                  n_samples, value, impurity)
    if !is_leaf(tree.binarytree, node_id)
        split_name = tree.features[tree.split_features[node_id]]
        split_val = tree.split_values[node_id]
        s *= @sprintf("; split: %s<=%.3f", split_name, split_val)
    end
    return s
end

function print_tree(tree::DecisionTreeClassifier)
    depths = find_depths(tree.binarytree)
    for (i, node) in enumerate(1:size(tree.binarytree))
        d = depths[node]
        s = @sprintf("%03d ", i)
        println(s, "-"^d, node_to_string(tree, node))
    end
    return
end

size(tree::DecisionTreeClassifier) = size(tree.binarytree)
get_max_depth(tree::DecisionTreeClassifier) = get_max_depth(tree.binarytree)

## --------------------- fitting --------------------- ##

function fit!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame)
    @assert size(Y, 2) == 1 "Output Y must be an m x 1 DataFrame"

    # set internal variables
    tree.n_features = size(X, 2)
    tree.n_classes = size(unique(Y), 1)
    tree.features = names(X)

    # fit
    split_node!(tree, X, Y, 0)

    # set attributes
    tree.feature_importances = feature_importance_impurity(tree)

    return
end

function count_classes(Y, n::Int)
    counts = zeros(n)
    for entry in eachrow(Y)
        counts[entry[1]] += 1.0
    end
    return counts
end

function set_defaults!(tree::DecisionTreeClassifier, Y::DataFrame)
    values = count_classes(Y, tree.n_classes)
    push!(tree.values, values)
    push!(tree.impurities, gini_score(values))
    push!(tree.split_features, nothing)
    push!(tree.split_values, nothing)
    push!(tree.n_samples, size(Y, 1))
    add_node!(tree.binarytree)

end

function split_node!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame, depth::Int)
    tree.num_nodes += 1
    node_id = tree.num_nodes
    set_defaults!(tree, Y)
    if tree.impurities[node_id] == 0.0
        return # only one class in this node
    end

    # random shuffling ensures a random variable is used if 2 splits are equal or if all features are used
    n_features_split = isnothing(tree.max_features) ? tree.n_features : min(tree.n_features, tree.max_features)
    features = randperm(tree.random_state, tree.n_features)[1:n_features_split]

    # make the split
    best_score = Inf
    for i in features
        best_score = find_better_split(i, X, Y, node_id, best_score, tree)
    end
    if best_score == Inf
        return # no split was made
    end

    # make children
    if isnothing(tree.max_depth) || (depth < tree.max_depth)
        x_split = X[:, tree.split_features[node_id]]
        lhs = x_split .<= tree.split_values[node_id]
        rhs = x_split .>  tree.split_values[node_id]
        set_left_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[lhs, :], Y[lhs, :], depth + 1)
        set_right_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[rhs, :], Y[rhs, :], depth + 1)
    end

    return
end

function find_better_split(feature_idx, X::DataFrame, Y::DataFrame, node_id::Int,
                           best_score::AbstractFloat, tree::DecisionTreeClassifier)
    x = X[:, feature_idx]

    n_samples = length(x)

    order = sortperm(x)
    x_sort, y_sort = x[order], Y[order, 1]

    rhs_count = count_classes(y_sort, tree.n_classes)
    lhs_count = zeros(tree.n_classes)

    xi, yi = zero(x_sort[1]), zero(y_sort[1]) # declare variables used in the loop (for optimisation purposes)
    for i in 1:(n_samples-1)
        global xi = x_sort[i]
        global yi = y_sort[i]
        lhs_count[yi] += 1.0; rhs_count[yi] -= 1.0
        if (xi == x_sort[i+1]) || (sum(lhs_count) < tree.min_samples_leaf)
            continue
        end
        if sum(rhs_count) < tree.min_samples_leaf
            break
        end
        # Gini impurity
        curr_score = (gini_score(lhs_count) * sum(lhs_count) + gini_score(rhs_count) * sum(rhs_count))/n_samples
        if curr_score < best_score
            best_score = curr_score
            tree.split_features[node_id] = feature_idx
            tree.split_values[node_id]= (xi + x_sort[i+1])/2
        end
    end
    return best_score
end

## --------------------- prediction --------------------- ##
"""
    predict_row(tree::DecisionTreeClassifier, xi<: DataFrameRow) => Vector{Int}

Returns the counts at the leaf node for a sample xi given a fitted DecisionTreeClassifier.
"""
function predict_row(tree::DecisionTreeClassifier, xi::T ) where T <: DataFrameRow
    next_node = 1
    while !is_leaf(tree.binarytree, next_node)
        left, right = get_children(tree.binarytree, next_node)
        next_node = xi[tree.split_features[next_node]] <= tree.split_values[next_node] ? left : right
    end
    return tree.values[next_node]
end

"""
    predict_batch(tree::DecisionTreeClassifier, X::DataFrame; node_id=1) => Matrix

Predict normalized weighting for each class for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done in batches -> all samples which follow the same path along the tree will be entered at the same time.
node_id is for internal use and should not be given as a parameter.
"""
function predict_batch(tree::DecisionTreeClassifier, X::DataFrame; node_id=1)
    # more complex and no speed increase
    if  (size(X, 1) == 0)
        return
    elseif is_leaf(tree.binarytree, node_id)
        counts = tree.values[node_id]
        return transpose(counts/sum(counts))
    end
    x_split = X[:, tree.split_features[node_id]]
    lhs = x_split .<= tree.split_values[node_id]
    rhs = x_split .>  tree.split_values[node_id]

    left, right = get_children(tree.binarytree, node_id)

    probs = zeros(nrow(X), tree.n_classes)
    if any(lhs)
        probs[lhs, :] .= predict_batch(tree, X[lhs, :], node_id=left)
    end
    if any(rhs)
        probs[rhs, :] .= predict_batch(tree, X[rhs, :], node_id=right)
    end
    return probs
end

"""
    predict_prob(tree::DecisionTreeClassifier, X::DataFrame) => Matrix

Predict normalized weighting for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done individually for each sample in X.
"""
function predict_prob(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs  = zeros(nrow(X), tree.n_classes)
    for (i, xi) in enumerate(eachrow(X))
        counts = predict_row(tree, xi)
        probs[i, :] .= counts/sum(counts)
    end
    return probs
end

"""
    predict(tree::DecisionTreeClassifier, X::DataFrame) => Matrix

Predict classes for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done individually for each sample in X.
"""
function predict(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs = predict_prob(tree, X)
    return mapslices(argmax, probs, dims=2)[:, 1]
end


## --------------------- feature importances --------------------- ##
"""
    nleaves(tree::DecisionTreeClassifier) => Int

Return the number of leaves in the DecisionTreeClassifier.
nleaves = size(binarytree) - nodes(binarytree)
"""
nleaves(tree::DecisionTreeClassifier) = nleaves(tree.binarytree)

"""
    feature_importance_impurity(tree::DecisionTreeClassifier) => Vector{Float64}

Calculate feature importance based on impurity.
For each feature, this is the weighted sum of the decrease in impurity that each node where that feature is used achieves.
The weights are the proportion of training samples present in each node out of the total samples.
"""
function feature_importance_impurity(tree::DecisionTreeClassifier)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    feature_importances = zeros(tree.n_features)
    total_samples = tree.n_samples[1]
    for node in 1:length(tree.impurities)
        if is_leaf(tree.binarytree, node)
            continue
        end
        spit_feature = tree.split_features[node]
        impurity = tree.impurities[node]
        n_samples = tree.n_samples[node]
        # calculate score
        left, right = get_children(tree.binarytree, node)
        lhs_gini = tree.impurities[left]
        rhs_gini = tree.impurities[right]
        lhs_count = tree.n_samples[left]
        rhs_count = tree.n_samples[right]
        score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
        # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
        feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    end
    # normalise
    feature_importances = feature_importances/sum(feature_importances)
    return feature_importances
end
