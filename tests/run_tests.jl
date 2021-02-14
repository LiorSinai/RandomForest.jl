#=
17 December 2020

Random Forests from scratch.
Unit tests
=#
include("../TreeEnsemble.jl")
using .TreeEnsemble
using Test

using CSV, DataFrames
using Random

println("Julia version: ", VERSION)
println("Running tests ...")


## ------------  test BinaryTree ------------ ##
@testset "BinaryTree Tests" begin

tree = BinaryTree()
add_node!(tree)
add_node!(tree)
add_node!(tree)
set_left_child!(tree, 1, 1)
set_left_child!(tree, 2, 2)
set_left_child!(tree, 3, 3)
set_right_child!(tree, 1, 4)
set_right_child!(tree, 2, 5)
@test tree.children_left == [1, 2, 3]
@test tree.children_right == [4, 5, -1]

end


## ------------  Full Classification ------------ ##
### load data
path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests/"
file_name = "tests/Iris_cleaned.csv"
target = "Species"

data = CSV.read(file_name, DataFrame)
X = select(data, Not(target))
Y = select(data, target)

max_features = 4
min_samples_leaf = 3
n_trees = 10

@testset "DecisionTreeClassifier Tests" begin
classifier = DecisionTreeClassifier(random_state=42,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf)

X_train, y_train, X_test, y_test = split_data(X, Y, rng=classifier.random_state, test_size=0.2)

fit!(classifier, X_train, y_train)
acc_train = score(classifier, X_train, y_train)
acc_test = score(classifier, X_test, y_test)
@test acc_train > 0.95
@test acc_test  > 0.93

n_samples = size(X, 1)
@test mapslices(sum, predict_prob(classifier, X), dims=[2])[:,1] ≈ ones(n_samples)

end #@testset "DecisionTreeClassifier Tests"
;


@testset "RandomForestClassifier Tests" begin
classifier = RandomForestClassifier(random_state=42,
                                    n_trees=n_trees,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=true,
                                    oob_score=true)

X_train, y_train, X_test, y_test = split_data(X, Y, rng=classifier.random_state, test_size=0.2)

fit!(classifier, X_train, y_train)
acc_train = score(classifier, X_train, y_train)
acc_test = score(classifier, X_test, y_test)
@test acc_train > 0.95
@test classifier.oob_score_ > 0.95
@test acc_test  > 0.95

n_samples = size(X, 1)
@test mapslices(sum, predict_prob(classifier, X), dims=[2])[:,1] ≈ ones(n_samples)

@test classifier.feature_importances ≈ [0.0, 0.0, 0.2652295143082535, 0.7347704856917466]
rng = MersenneTwister(2)
fi =  perm_feature_importance(classifier, X_train, y_train, n_repeats=10, random_state=rng)
@test fi[:means] ≈ [0.0, 0.0, 0.07416666666666666, 0.54]

end #@testset "RandomForestClassifier Tests"
;
