# RandomForest-jl
# Random Forest Classifier from Scratch

## Overview

A custom random forest implementation. See also my Python based implementation at [github.com/LiorSinai/randomForests](https://github.com/LiorSinai/randomForests).

## Type Tree

`Any`

\-  `AbstractClassifier`

\-\- `RandomForestClassifier{T}`

\-\- `DecisionTreeClassifier{T}`

## AbstractClassifier

The `AbstractClassifier` has the following methods defined for it:

- `score(classfier::AbstractClassifier, X::DataFrame, Y::DataFrame)`: returns the fraction of correct predictions of X for y.
- `perm_feature_importance(classifier::AbstractClassifier,  X::DataFrame, Y::DataFrame; n_repeats=10, random_state=Random.GLOBAL_RNG)`: the feature importance based on random permutations of each feature column.

## RandomForestClassifier
The class is created with:
`forest = RandomForestClassifier{T}([n_trees=100], [max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG], [bootstrap=true], [oob_score=false])`

The type `T` specifies the type of values in the dataframe. Leaving it out will default to `Float64`.

The parameters are:
- `n_trees::Int`: number of trees (estimators).
- `random_state::Union{AbstractRNG, Int}`: a random number genetator or a seed for a `Random.MersenneTwister`.
- `max_features::Union{Int, Nothing}`: maximum number of features to randomly select from per split.
- `max_depth::Int`: stop splitting after reaching this depth.
- `min_samples_leaf::Int`: the mininum number of samples allowed in a leaf.
- `bootstrap::Bool`: use a random subset of samples per tree.
- `oob_score::Bool` (only used if `bootstrap=true`): calculate the out-of-bag score. This is the mean accuracy of the predictions made for each sample using
only the trees that were _not_ trained on that sample.

Let the _k=sample_size_ and _n=total_samples_.
Then on average with replacement, n(1-1/n)^k ~ exp(-n/k) samples will not be used in each tree. This is 36.8% of samples per tree if _k=n_.
This is a significant portion of samples. Therefore the out-of-bag samples form a useful proxy validation set.

It has the following external methods:
- `fit!(forest::RandomForestClassifier, X::DataFrame, Y::DataFrame)`: fit the data to the random forest classifier. Y can have multiple classes.
- `predict(forest::RandomForestClassifier, X::DataFrame)`: returns the of predicted classes y for the independent variable X. Predictions are made using (soft) majority voting between trees. That is, the predicted has the maximum sum of probabilities.
- `predict_prob(forest::RandomForestClassifier, X::DataFrame)`: returns the predicted proportion (probability) in the leaf nodes for each sample in X summed over all trees.

It has the following attributes:
- `trees::Vector{DecisionTreeClassifier{T}}`: list of DecisionTreeClassifier.
- `n_features::Int`: the nubmer of features of the training dataframe X.
- `n_class::Int`: the nubmer of classes of the training dataframe Y.
- `features::Vector{String}`: the names of the features in the training dataframe X.
- `feature_importances::Union{Vector{Float64}, Nothing}`: the feature importance calculated per feature as the sum of the change in impurity per node where that feature splits the node,
weighted by the fraction of samples used in that node.
Unlike the permutation importance, this is independent of the input data.
- `oob_score_::Union{Float64, Nothing}`: the mean out-of-bag score.

## DecisionTree
A binary decision tree. It is based off Scikit-learn's [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
 The tree is encoded as a set of parallel lists for children_left and children_right.

The class is created with:
`tree = DecisionTreeClassifier{T}([max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG])`.

The type `T` specifies the type of values in the dataframe. Leaving it out will default to `Float64`.

The parameters are:
- `random_state::Union{AbstractRNG, Int}`: a random number genetator or a seed for a `Random.MersenneTwister`.
- `max_features::Union{Int, Nothing}`: maximum number of features to randomly select from per split.
- `max_depth::Int`: stop splitting after reaching this depth.
- `min_samples_leaf::Int`: the mininum number of samples allowed in a leaf.

It has the following external methods:
- `predict_prob(tree::DecisionTreeClassifier, X::DataFrame)`: returns the predicted proportion (probability) in the leaf nodes for each sample in X.
- `predict_row(tree::DecisionTreeClassifier, xi::T ) where T <: DataFrameRow`: returns the counts at the leaf node for a single sample.
- `predict(tree::DecisionTreeClassifier, X::DataFrame)`: returns the of predicted classes y for the independent variable X.
- `node_to_string(tree::DecisionTreeClassifier, node_id::Int)`: returns a formatted string of the get_info(node_id) data.
- `print_tree(tree::DecisionTreeClassifier)`: print the tree to the output screen.
- `size(tree::DecisionTreeClassifier)`: returns the number of nodes (+leaves) in the binary tree. Extends `Base.size`.
- `get_max_depght(tree::DecisionTreeClassifier)`: Calculate the maximum depth of the tree.


It has the following attributes:
- `binaryTree::BinaryTree`: A binary tree encoded as a set of parallel lists for children_left and children_right.
- `n_samples::Vector{Int}`: number of samples in this node.
- `impurities::Vector{Float64}`: Gini impurity of each node. Sklearn equivalent.
- `values::Vector{Vector{Float64}}`: count per class in each node.
- `split_features::Vector{Union{Int, Nothing}}`: feature used to split each node.
- `split_values::Vector{Union{T, Nothing}}`: value used to split each node. `T` is only used here.
- `n_features::Int`: the nubmer of features of the training dataframe X.
- `n_class::Int`: the nubmer of classes of the training dataframe Y.
- `features::Vector{String}`: the names of the features in the training dataframe X.
- `feature_importances::Union{Vector{Float64}, Nothing}`: the feature importance calculated per feature as the sum of the change in impurity per node where that feature splits the node,
weighted by the fraction of samples used in that node.

## Test data sets

Two test sets are used from Kaggle:
- [Bank_Loan_Classification](https://www.kaggle.com/sriharipramod/bank-loan-classification/): Binary classification problem of approved loads. 5000 entries and 14 features.
This is an easy problem. The feature Income is a high valued predictor. Using Income>100 as a benchmark model achieves a 83.52% accuracy over the whole data set.
The random forest achieves up to 99.10% accuracy on test datasets.
- [Iris Species](https://www.kaggle.com/uciml/iris): 3-class classification problem of iris species based on plant dimensions. 150 entries and 3 features.
The random forest achieves 96%-100% accuracy on test datasets.

## Dependencies

`DataFrame`, `CSV`, `Plots`, `Random`
