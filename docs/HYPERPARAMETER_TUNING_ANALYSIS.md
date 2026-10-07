# Hyperparameter Tuning Analysis

## 1. Baseline Model

**Model:**  
DecisionTreeClassifier

**CV F1 Macro:**  
0.9663

**Test Accuracy:**  
0.9000

The baseline Decision Tree was used as a reference model before applying
hyperparameter tuning.

## 2. Grid Search

**Model:**  
RandomForestClassifier

**Total combinations:**  
72

**Cross-validation:**  
5-fold

**Total fits:**  
360

**Best CV F1 Macro:**  
0.9663

**Test Accuracy:**  
0.9667

Grid Search exhaustively evaluated all 72 combinations of the specified
hyperparameter values using 5-fold cross-validation.

## 3. Random Search

**Model:**  
RandomForestClassifier

**Number of iterations:**  
72

**Cross-validation:**  
5-fold

**Total fits:**  
360

**Best CV F1 Macro:**  
0.9663

**Test Accuracy:**  
0.9667

Random Search evaluated 72 randomly selected hyperparameter combinations
using 5-fold cross-validation.

## 4. Comparison

Grid Search evaluates all combinations in the specified hyperparameter grid.

Random Search evaluates a selected number of combinations from the search
space.

In this experiment, Grid Search required 360 model fits, while Random Search
also required 360 model fits because 72 iterations were used with 5-fold
cross-validation.

The baseline Decision Tree achieved 90.00% test accuracy.

Both tuned Random Forest approaches achieved 96.67% test accuracy,
representing an improvement of 6.67 percentage points over the baseline.

Both Grid Search and Random Search achieved the same best CV F1 Macro score
of 0.9663 and the same test accuracy of 0.9667.

For larger hyperparameter search spaces, Random Search can be more efficient
when a smaller number of iterations is used because it can explore a broad
search space without evaluating every possible combination.