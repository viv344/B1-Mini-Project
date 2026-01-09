# B1 Project - Logistic Regression for Binary Classification
## Complete Implementation and Analysis Summary

---

## ALL TASKS COMPLETED (Section 3.2)

### **Implementation (Section 3.1)** ✓
All core functions implemented:
- ✓ `grad_descent()` - Gradient descent optimization with loss tracking
- ✓ `log_regr()` - Logistic regression prediction (sigmoid function)
- ✓ `mean_logloss()` - Mean log-loss cost function
- ✓ `classif_error()` - Classification error percentage
- ✓ `create_features_for_poly()` - Polynomial feature generation

### **Analysis Tasks (Section 3.2)** ✓

#### **Task 1: Hyper-parameter Configuration** ✓
**Script:** `hyperparameter_tuning.py`
**Results:**
- Tested: n_iters = {100, 500, 1000, 10000} × λ = {0.01, 0.1, 0.5, 1.0}
- **Best configuration:** λ = 0.1, n_iters = 10000
- **Validation error:** 3.00%
- Generated: `hyperparameter_tuning_results.png`

#### **Task 2: Convergence Analysis** ✓
**Script:** `plot_convergence.py`
**Results:**
- Initial loss: 0.6931 (= ln(2), as expected!)
- Final loss: 0.0977
- **Loss reduction: 85.91%**
- Model properly converged
- Generated: `convergence_analysis.png`, `convergence_comparison_learning_rates.png`

#### **Task 3: Runtime vs Quality Trade-off** ✓
**Script:** `analyze_runtime_quality.py`
**Results:**
- Runtime scales linearly with n_iters
- λ affects convergence speed but not runtime
- **Best balance:** λ = 0.1-0.5 with n_iters = 1000-5000
- Generated: `runtime_quality_analysis.png`

#### **Task 4: Repeated Experiments** ✓
**Script:** `repeated_experiments.py`
**Results:**
- Infrastructure for 10-20 repetitions per configuration
- Averaged results: Training 3.05% ± 0.82%, Validation 3.58% ± 0.35%
- Accounts for randomness in data sampling

#### **Task 5: Polynomial Degree Comparison** ✓
**Script:** `compare_polynomial_degrees.py`
**Results:** (Averaged over 20 repetitions)

| Degree | Train Error | Val Error | Overfitting Gap |
|--------|-------------|-----------|-----------------|
| 1      | 9.54% ± 1.48% | 9.46% ± 0.50% | -0.08% |
| 2      | 5.08% ± 1.19% | 5.17% ± 0.37% | 0.09% |
| **3**  | **2.95% ± 0.71%** | **3.41% ± 0.39%** | **0.46%** ✓ |
| 4      | 3.01% ± 0.70% | 3.68% ± 0.34% | 0.67% |
| 5      | 2.71% ± 0.93% | 3.46% ± 0.43% | 0.75% |

**Best:** Degree 3 (3.41% validation error)
Generated: `polynomial_degree_comparison.png`

#### **Task 6: Training Set Size Analysis** ✓
**Script:** `compare_training_sizes.py`
**Results:** (Averaged over 20 repetitions)

| Samples | Train Error | Val Error | Overfitting Gap |
|---------|-------------|-----------|-----------------|
| 50      | 1.00% ± 1.48% | 5.35% ± 1.33% | **4.35%** |
| 100     | 1.75% ± 1.51% | 4.19% ± 0.94% | **2.44%** |
| 200     | 3.33% ± 1.25% | 3.85% ± 0.45% | 0.52% ✓ |
| **400** | **3.02% ± 0.92%** | **3.50% ± 0.43%** | **0.48%** ✓ |

**Key finding:** Severe overfitting with small datasets! 400 samples optimal.
Generated: `training_size_comparison.png`

#### **Task 7: Best Model Analysis** ✓
**Script:** `best_model_analysis.py`
**Best Model:**
- **Configuration:** Degree 3, λ = 0.1, n_iters = 10000
- **Performance:** 3.88% validation error, 3.50% training error
- **Parameters:** 10 optimal θ values learned
- **Expected:** YES - Non-linear data requires non-linear model

**Optimal θ parameters:**
```
[ 0.789  1.115 -0.514  3.837  0.815 -2.978  1.075  2.553 -0.817 -3.431]
```

Generated: `best_model_decision_boundary.png`, `linear_vs_nonlinear_comparison.png`

#### **Task 8: Numerical Issues** ✓
**Script:** `numerical_issues_report.py`
**Issues Resolved:**
1. **Log of zero:** Used epsilon clipping (1e-15)
2. **Sigmoid overflow:** NumPy handles gracefully
3. **Gradient precision:** Proper scaling with (1/n)
4. **Dimension compatibility:** Explicit reshaping to column vectors

---

## Generated Files Summary

### **Code Files:**
- `main_project_skeleton.py` - Core implementation
- `create_data.py` - Data generation (provided, not modified)
- `hyperparameter_tuning.py` - Task 1
- `plot_convergence.py` - Task 2
- `analyze_runtime_quality.py` - Task 3
- `repeated_experiments.py` - Task 4 infrastructure
- `compare_polynomial_degrees.py` - Task 5
- `compare_training_sizes.py` - Task 6
- `best_model_analysis.py` - Task 7
- `numerical_issues_report.py` - Task 8

### **Generated Plots:**
1. `hyperparameter_tuning_results.png` - Task 1 results
2. `convergence_analysis.png` - Task 2 convergence (4 panels)
3. `convergence_comparison_learning_rates.png` - Task 2 λ comparison
4. `runtime_quality_analysis.png` - Task 3 trade-offs
5. `polynomial_degree_comparison.png` - Task 5 analysis
6. `training_size_comparison.png` - Task 6 learning curves
7. `best_model_decision_boundary.png` - Task 7 visualization
8. `linear_vs_nonlinear_comparison.png` - Task 7 comparison

---

## 🎓 Key Insights for Report

1. **Non-linear models essential:** Linear model (9.46% error) vs Degree 3 (3.41% error)

2. **Optimal complexity:** Degree 3 balances bias-variance tradeoff

3. **Data size matters:** 50 samples → 4.35% overfitting gap vs 400 samples → 0.48% gap

4. **Hyperparameter tuning critical:** λ=0.1, n_iters=10000 gives best results

5. **Convergence validation:** 85.9% loss reduction confirms proper training

6. **Numerical stability:** Epsilon clipping and proper scaling prevent errors

---

## Final Performance Summary

**Best Model Configuration:**
- Polynomial degree: 3
- Learning rate: 0.1
- Iterations: 10000
- Training samples: 400

**Performance:**
- Validation error: 3.41% (averaged), 3.88% (single run)
- Training error: 2.95% (averaged), 3.50% (single run)
- Overfitting gap: 0.46% (minimal!)

**Improvement over baseline:**
- 64% better than linear model (9.46% → 3.41%)
- 35% better than 50-sample training (5.35% → 3.41%)

---

## All Section 3.2 Tasks Complete!

Ready for report writing with:
- ✓ Tables of results
- ✓ Visualizations
- ✓ Analysis and discussions
- ✓ Optimal parameters
- ✓ Numerical stability notes

**Project Status: COMPLETE**
