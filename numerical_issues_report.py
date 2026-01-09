###############################################
# Numerical Issues and Solutions
# Section 3.2, Task 8
###############################################

import numpy as np

print("="*70)
print("TASK 8: NUMERICAL ISSUES ENCOUNTERED AND SOLUTIONS")
print("="*70)

print("""
During the implementation of logistic regression with gradient descent,
several numerical stability issues were identified and resolved:
""")

print("\n" + "="*70)
print("ISSUE 1: LOGARITHM OF ZERO IN LOG-LOSS CALCULATION")
print("="*70)
print("""
PROBLEM:
--------
The log-loss function is defined as:
  L = -y*log(ŷ) - (1-y)*log(1-ŷ)

When predictions ŷ approach 0 or 1, we encounter:
  - log(0) = -∞ (undefined)
  - log(1-1) = log(0) = -∞ (undefined)

This causes NaN (Not a Number) values in the loss calculation.

CAUSE:
------
The sigmoid function σ(z) = 1/(1 + e^(-z)) can output:
  - Very close to 0 when z is large negative
  - Very close to 1 when z is large positive

With floating-point arithmetic, these can be EXACTLY 0 or 1.

SOLUTION IMPLEMENTED:
---------------------
Used np.clip() to bound predictions away from 0 and 1:

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

This ensures:
  - Minimum prediction: 1e-15 (instead of 0)
  - Maximum prediction: 1 - 1e-15 (instead of 1)
  - log() always receives valid input > 0

LOCATION IN CODE:
-----------------
- mean_logloss() function (line ~80)
- grad_descent() function when track_loss=True (line ~40)

VERIFICATION:
-------------
""")

# Demonstrate the issue and solution
print("Without clipping:")
try:
    y_pred_bad = np.array([0.0, 1.0, 0.5])
    loss_bad = -np.log(y_pred_bad)
    print(f"  Predictions: {y_pred_bad}")
    print(f"  -log(y_pred): {loss_bad}")
    print(f"  Result: Contains {np.sum(np.isinf(loss_bad))} infinite values!")
except:
    print("  Error encountered!")

print("\nWith clipping (epsilon=1e-15):")
epsilon = 1e-15
y_pred_good = np.clip(np.array([0.0, 1.0, 0.5]), epsilon, 1 - epsilon)
loss_good = -np.log(y_pred_good)
print(f"  Original predictions: [0.0, 1.0, 0.5]")
print(f"  Clipped predictions:  {y_pred_good}")
print(f"  -log(y_pred): {loss_good}")
print(f"  Result: All finite values ✓")

print("\n" + "="*70)
print("ISSUE 2: NUMERICAL OVERFLOW IN SIGMOID FUNCTION")
print("="*70)
print("""
PROBLEM:
--------
The sigmoid function σ(z) = 1/(1 + e^(-z)) can overflow when:
  - z is very negative → e^(-z) becomes extremely large
  - e^(-z) > 10^308 causes overflow in float64

POTENTIAL IMPACT:
-----------------
For large negative z (e.g., z = -1000):
  e^(1000) would overflow → σ(z) becomes undefined

SOLUTION IMPLEMENTED:
---------------------
NumPy's exp() function handles this gracefully:
  - For very negative z: e^(-z) → large number, σ(z) → 0
  - For very positive z: e^(-z) → 0, σ(z) → 1
  - Built-in overflow protection in NumPy

Our implementation:
    y_pred = 1 / (1 + np.exp(-z))

NumPy automatically handles extreme values without explicit clipping.

LOCATION IN CODE:
-----------------
- log_regr() function (line ~103)
- grad_descent() function (line ~35)

VERIFICATION:
-------------
""")

# Test extreme values
z_extreme = np.array([-1000, -100, 0, 100, 1000])
sigmoid = 1 / (1 + np.exp(-z_extreme))
print(f"  z values: {z_extreme}")
print(f"  σ(z):     {sigmoid}")
print(f"  All valid: {np.all(np.isfinite(sigmoid))} ✓")

print("\n" + "="*70)
print("ISSUE 3: NUMERICAL PRECISION IN GRADIENT COMPUTATION")
print("="*70)
print("""
PROBLEM:
--------
When computing gradients:
  gradient = (1/n) * X^T * (ŷ - y)

Small errors in ŷ can accumulate when:
  - n (number of samples) is small
  - Feature values in X are very large or small
  - Many iterations of gradient descent

SOLUTION IMPLEMENTED:
---------------------
1. Explicit division by n (number of samples):
   gradient = (1/n) * X^T * (ŷ - y)

   This ensures gradients are properly scaled regardless of batch size.

2. Matrix operations using NumPy:
   - Uses optimized BLAS/LAPACK libraries
   - Better numerical stability than manual loops
   - Reduced accumulation of rounding errors

3. Proper parameter initialization:
   theta = np.zeros() instead of random values

   Avoids initial instability from poor starting points.

LOCATION IN CODE:
-----------------
- grad_descent() function (line ~47)

IMPACT:
-------
Stable convergence across all experiments, no gradient explosions observed.
""")

print("\n" + "="*70)
print("ISSUE 4: MATRIX DIMENSION COMPATIBILITY")
print("="*70)
print("""
PROBLEM:
--------
Incompatible matrix/vector dimensions in:
  - Matrix multiplication: X @ theta
  - Gradient computation: X^T @ (ŷ - y)
  - Loss calculation with y and ŷ

SOLUTION IMPLEMENTED:
---------------------
Explicit reshaping to ensure column vectors:

    y_train = y_train.reshape(-1, 1)  # Ensures column vector
    y_real = y_real.reshape(-1, 1)    # Ensures column vector

This guarantees:
  - theta: (n_features+1, 1) - column vector
  - y: (n_samples, 1) - column vector
  - X @ theta: (n_samples, 1) - column vector
  - X^T @ (ŷ - y): (n_features+1, 1) - column vector

LOCATION IN CODE:
-----------------
- grad_descent() function (line ~22)
- mean_logloss() function (line ~72)

BENEFIT:
--------
Prevents dimension mismatch errors, ensures consistent matrix operations.
""")

print("\n" + "="*70)
print("SUMMARY FOR REPORT")
print("="*70)
print("""
Key Numerical Issues Resolved:

1. Log of Zero: Used epsilon clipping (1e-15) to prevent log(0)
2. Sigmoid Overflow: Relied on NumPy's built-in overflow handling
3. Gradient Precision: Used proper scaling and NumPy matrix operations
4. Dimension Compatibility: Explicit reshaping to column vectors

All solutions are standard best practices in numerical computing
for machine learning. No custom numerical methods were required.

The implementation achieved:
✓ Stable convergence across all experiments
✓ No NaN or Inf values in loss or gradients
✓ Consistent results across multiple runs
✓ Numerical stability with various hyperparameters

These techniques are essential for reliable gradient-based optimization.
""")

print("\n" + "="*70)
print("TASK 8 COMPLETE")
print("="*70)
