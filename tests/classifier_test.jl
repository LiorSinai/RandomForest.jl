#=
8 December 2020

Random Forests from scratch. Redo of Python code

Test code

=#

include("../TreeEnsemble.jl")
using .TreeEnsemble

#debugging -> direct imports, makes it easier to modify functions during testing
## produces slower cose the code
# include("../Utilities.jl")
# include("../Classifier.jl")
# include("../DecisionTree.jl")
# include("../RandomForest.jl")

using CSV, DataFrames
using Printf
using Statistics

## ------------  test BinaryTree ------------ ##

tree = BinaryTree()
add_node!(tree)
add_node!(tree)
add_node!(tree)
set_left_child!(tree, 1, 1)
set_left_child!(tree, 2, 2)
set_left_child!(tree, 3, 3)
set_right_child!(tree, 1, 4)
set_right_child!(tree, 2, 5)
@assert tree.children_left == [1, 2, 3]
@assert tree.children_right == [4, 5, -1]

## -------------- load data  -------------- ##

path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests/"
#file_name = "tests/Iris_cleaned.csv"
# target = "Species"
# max_features = 4
# n_trees = 10
# min_samples_leaf = 1

file_name = "tests/UniversalBank_cleaned.csv"
target = "Personal Loan"
max_features = 3
n_trees = 20
min_samples_leaf = 3

data = CSV.read(file_name, DataFrame)
X = select(data, Not(target))
Y = select(data, target)

## -------------- fit data  -------------- ##
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42, n_trees=n_trees, bootstrap=true, oob_score=true,
                             max_features=max_features, min_samples_leaf=min_samples_leaf)

classifier = rfc

X_train, y_train, X_test, y_test = split_data(X, Y, rng=classifier.random_state, test_size=0.2)

println()
print("fitting time           "); @time fit!(classifier, X_train, y_train)
print("prediction time (train)"); @time acc_train = score(classifier, X_train, y_train)
print("prediction time (test) "); @time acc_test = score(classifier, X_test, y_test)


println()
@printf("train accuracy: %.2f%%\n", acc_train*100)
if hasproperty(classifier, :oob_score_) && !isnothing(classifier.oob_score_)
    @printf("obb accuracy:   %.2f%%\n", classifier.oob_score_*100)
end
@printf("test accuracy:  %.2f%%\n", acc_test*100)

println()
nleaves_ = nleaves(classifier)
@printf("nleaves range, average: %d-%d, %.2f\n",
        minimum(nleaves_),  maximum(nleaves_), mean(nleaves_))

println()
# confusion matrix
y_actual = y_test[:, 1]
y_pred =  predict(classifier, X_test)
C = confusion_matrix(y_actual, y_pred)
println("Confusion matrix:")
display(C);
if size(C, 1) == 2
    recall, prec, f1 = calc_f1_score(C)
    @printf("recall, precision, F1: %.2f%%, %.2f%%, %.4f\n",
            recall*100, prec*100, f1)
end

println()
fi = classifier.feature_importances
#print("perm_feature_importance time")
#@time fi = perm_feature_importance(classifier, X_train, y_train, n_repeats=10, random_state=classifier.random_state)[:means]
order = sortperm(fi, rev=true)
println("Feature importances")
for col_val in zip(names(X_train)[order], fi[order])
    col, val = col_val
    @printf("%-15s %.4f\n",col, val)
end

#println()
#print_tree(rfc.trees[3])
