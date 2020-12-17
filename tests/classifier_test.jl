#=
8 December 2020

Random Forests from scratch. Redo of Python code

Test code

=#

using Revise
include("../TreeEnsemble.jl")

if !isdefined(Main, :TreeEnsemble)
    using .TreeEnsemble
end

using Plots

# call during debugging:
# Revise.track("Utilities.jl")
# Revise.track("Classifier.jl")
# Revise.track("DecisionTree.jl")
# Revise.track("RandomForest.jl")

using CSV, DataFrames
using Printf
using Statistics

## -------------- load data  -------------- ##

path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests/"

file_name = "tests/Iris_cleaned.csv"
target = "Species"
max_features = 4
n_trees = 10
min_samples_leaf = 3

# file_name = "tests/UniversalBank_cleaned.csv"
# target = "Personal Loan"
# max_features = 3
# n_trees = 20
# min_samples_leaf = 3

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
if hasproperty(classifier, :trees)
    max_depths = [get_max_depth(tree) for tree in classifier.trees]
else
    max_depths = [get_max_depth(classifier)]
end
@printf("nleaves range, average: %d-%d, %.2f\n",
        minimum(nleaves_),  maximum(nleaves_), mean(nleaves_))
@printf("max depth range, average: %d-%d, %.2f\n",
        minimum(max_depths),  maximum(max_depths), mean(max_depths))


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
fi1 = classifier.feature_importances
print("perm_feature_importance time")
@time fi_perm = perm_feature_importance(classifier, X_train, y_train, n_repeats=10, random_state=classifier.random_state)
fi2 = fi_perm[:means]
order = sortperm(fi2, rev=true)
println("Feature importances")
for col_val in zip(names(X_train)[order], fi1[order])
    col, val = col_val
    @printf("%-15s %.4f\n",col, val)
end

order = sortperm(fi2)
fi2 ./= sum(fi2)

width = 0.4
yticks2 = (1+width/2):1:(length(fi2)+width/2)
yticks  = (1-width/2):1:(length(fi2)-width/2)

b1 = bar(yticks2, fi2[order], label="permutation", orientation = :horizontal, bar_width=width, yerr=fi_perm[:stds])
bar!(yticks, fi1[order], label="impurity", orientation = :horizontal, bar_width=width)
plot!(
    yticks=(1:classifier.n_features, classifier.features[order]),
    xlabel = "relative feature importance score",
    ylabel = "feature",
    title = "Feature importances",
    label = "impurity", legend=(0.85, 0.8)
    )

display(b1)
#savefig(b1, "UniversalBank_feature_importances_jl")

# println()
# print_tree(rfc.trees[1])
