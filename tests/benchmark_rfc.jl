#=
14 December 2020

Random Forests from scratch. Redo of Python code

Benchmark tests

=#

include("../TreeEnsemble.jl")
using .TreeEnsemble

using CSV, DataFrames
using Printf
using Statistics

## -------------- load data  -------------- ##

path = "C:/Users/sinai/Documents/Projects/Julia projects/RandomForestClassifier-jl/tests/"
file_name = "tests/UniversalBank_cleaned.csv"
target = "Personal Loan"

data = CSV.read(file_name, DataFrame)
X = select(data, Not(target))
Y = select(data, target)

## -------------- fit data  -------------- ##
#parameters
max_features = 3
n_trees = 20
min_samples_leaf = 3

classifier = RandomForestClassifier(random_state=42, n_trees=n_trees, bootstrap=true, oob_score=true,
                                    max_features=max_features, min_samples_leaf=min_samples_leaf)

X_train, y_train, X_test, y_test = split_data(X, Y, rng=42, test_size=0.2)

n_trials = 10
times = zeros(1 + n_trials)

t = @timed fit!(classifier, X_train, y_train)
times[end] = t.time # compile time

for i in 1:n_trials
    local t = @timed fit!(classifier, X_train, y_train)
    times[i] = t.time
end

μ_jl = mean(times[1:end-1])
σ_jl = std(times[1:end-1])

# from python
μ_sk = 0.04737
σ_sk = 0.00598
μ_py = 7.46414
σ_py = 0.26655

println()
@printf("%d trials\n", n_trials)
@printf("First time: %.5fs\n", times[end])
@printf("Julia time: %.5fs ± %.5fs\n", μ_jl, σ_jl)
@printf("ratio SciKit:Julia : 1:%.4f, Julia:Python, 1:%.4f\n", μ_jl/μ_sk, μ_py/μ_jl)


# 10 random seeds
seeds = [2322, 64802, 8102, 95, 5744, 9336,  104, 6532, 6197, 549272]
n_seeds = length(seeds)

accuracy_jl = zeros(n_seeds)
for (i, seed) in enumerate(seeds)
    local X_train, y_train, X_test, y_test = split_data(X, Y, rng=seed, test_size=0.2)
    fit!(classifier, X_train, y_train)
    accuracy_jl[i] = score(classifier, X_test, y_test)
end

μ_jl = mean(accuracy_jl) * 100
σ_jl =  std(accuracy_jl) * 100

# from python
μ_sk = 98.43
σ_sk = 0.58
μ_py = 98.56
σ_py = 0.46
println()
@printf("%d trials\n", n_seeds)
@printf("Julia accuracy: %.2f%% ± %.2f%%\n", μ_jl, σ_jl)
