#=
12  December 2020

TreeEnsemble. Redo of Python code
Includes
- BinaryTree, with nodes/leaves stored in 2 parallel arrays in preorder
- AbstractClassifier
- DecisionTreeClassifier, which contains 1 BinaryTree
- RandomForestClassifier, which contains a vector of DecisionTreeClassifier


Sources
- https://course18.fast.ai/lessonsml1/lesson5.html
- https://github.com/bensadeghi/DecisionTree.jl

=#

module TreeEnsemble

export  AbstractClassifier, predict, score, fit!, perm_feature_importance,
        # Binary tree
        BinaryTree, add_node!, set_left_child!, set_right_child!, get_children,
        is_leaf, nleaves, find_depths, get_max_depth,
        # Decision Tree Classifier
        DecisionTreeClassifier, predict_row, predict_batch, predict_prob,
        feature_importance_impurity, print_tree, node_to_string,
        # Random Forest Classifier
        RandomForestClassifier,
        # utilities
        check_random_state, split_data, confusion_matrix, calc_f1_score,
        get_methods_with

include("Utilities.jl")
include("Classifier.jl")
include("DecisionTree.jl")
include("RandomForest.jl")

end
