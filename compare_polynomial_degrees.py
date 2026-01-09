###############################################
# Compare Polynomial Degrees (1, 2, 3, 4, 5)
# Section 3.2, Task 5
###############################################

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/vivek/Documents/B1 Mini Project')
from repeated_experiments import repeat_experiments

print("="*70)
print("TASK 5: COMPARING POLYNOMIAL DEGREES (1, 2, 3, 4, 5)")
print("="*70)

# Best hyperparameters from Task 1
best_lr = 0.1
best_n_iters = 10000
n_repetitions = 20
degrees_to_test = [1, 2, 3, 4, 5]

print(f"\nConfiguration:")
print(f"  Learning rate (λ): {best_lr}")
print(f"  Iterations: {best_n_iters}")
print(f"  Repetitions per degree: {n_repetitions}")
print(f"  Training samples: 400")
print(f"  Validation samples: 4000")
print()

# Store results for all degrees
all_results = {}

for degree in degrees_to_test:
    print(f"\n{'='*70}")
    print(f"Testing Polynomial Degree: {degree}")
    print(f"{'='*70}")

    results = repeat_experiments(
        degree_poly=degree,
        lr=best_lr,
        n_iters=best_n_iters,
        n_repetitions=n_repetitions
    )

    all_results[degree] = results

    print(f"\nResults for Degree {degree}:")
    print(f"  Train Error: {results['train_error_mean']:.2f}% ± {results['train_error_std']:.2f}%")
    print(f"  Val Error:   {results['val_error_mean']:.2f}% ± {results['val_error_std']:.2f}%")

# Create results table
print("\n" + "="*70)
print("RESULTS TABLE - AVERAGED OVER {} REPETITIONS".format(n_repetitions))
print("="*70)

print(f"\n{'Degree':<10} {'Train Error (%)':<20} {'Val Error (%)':<20} {'Overfitting Gap':<15}")
print("-" * 65)

for degree in degrees_to_test:
    r = all_results[degree]
    gap = r['val_error_mean'] - r['train_error_mean']
    print(f"{degree:<10} {r['train_error_mean']:>7.2f} ± {r['train_error_std']:<7.2f} "
          f"{r['val_error_mean']:>7.2f} ± {r['val_error_std']:<7.2f} {gap:>13.2f}%")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

best_val_degree = min(degrees_to_test, key=lambda d: all_results[d]['val_error_mean'])
best_train_degree = min(degrees_to_test, key=lambda d: all_results[d]['train_error_mean'])

print(f"\nBest Validation Performance:")
print(f"  Degree: {best_val_degree}")
print(f"  Validation Error: {all_results[best_val_degree]['val_error_mean']:.2f}%")
print(f"  Training Error: {all_results[best_val_degree]['train_error_mean']:.2f}%")

print(f"\nBest Training Performance:")
print(f"  Degree: {best_train_degree}")
print(f"  Training Error: {all_results[best_train_degree]['train_error_mean']:.2f}%")
print(f"  Validation Error: {all_results[best_train_degree]['val_error_mean']:.2f}%")

# Overfitting analysis
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)

print(f"\n{'Degree':<10} {'Overfitting Gap (%)':<25} {'Interpretation':<30}")
print("-" * 65)

for degree in degrees_to_test:
    r = all_results[degree]
    gap = r['val_error_mean'] - r['train_error_mean']

    if gap < 0.5:
        interpretation = "No overfitting"
    elif gap < 1.5:
        interpretation = "Minimal overfitting"
    elif gap < 3.0:
        interpretation = "Moderate overfitting"
    else:
        interpretation = "Significant overfitting!"

    print(f"{degree:<10} {gap:>10.2f}% {'':<14} {interpretation:<30}")

print("\n" + "="*70)
print("KEY OBSERVATIONS FOR REPORT")
print("="*70)
print("""
1. BEST MODEL:
   - Model with best validation performance should be chosen
   - This represents how well the model generalizes to new data

2. OVERFITTING PATTERNS:
   - Overfitting occurs when training error << validation error
   - Higher degree polynomials are more prone to overfitting
   - They have more parameters and can "memorize" training data
   - But this hurts performance on new validation data

3. MODEL COMPLEXITY TRADE-OFF:
   - Too simple (degree 1): Underfitting, high bias
     → Cannot capture complex non-linear decision boundary
   - Too complex (high degree): Overfitting, high variance
     → Fits training noise, doesn't generalize well
   - Optimal complexity: Balances bias and variance

4. LINEAR VS NON-LINEAR:
   - Degree 1 (linear) likely performs worst
   - The data has non-linear structure (multi-modal Gaussians)
   - Non-linear models (degree 2-4) should perform better
   - Very high degrees (5+) may overfit with limited data
""")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training vs Validation Error
ax1 = axes[0, 0]
train_means = [all_results[d]['train_error_mean'] for d in degrees_to_test]
train_stds = [all_results[d]['train_error_std'] for d in degrees_to_test]
val_means = [all_results[d]['val_error_mean'] for d in degrees_to_test]
val_stds = [all_results[d]['val_error_std'] for d in degrees_to_test]

ax1.errorbar(degrees_to_test, train_means, yerr=train_stds, marker='o',
             label='Training Error', linewidth=2, capsize=5)
ax1.errorbar(degrees_to_test, val_means, yerr=val_stds, marker='s',
             label='Validation Error', linewidth=2, capsize=5)
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Classification Error (%)')
ax1.set_title('Training vs Validation Error\n(Error bars show ± 1 std)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(degrees_to_test)

# Plot 2: Overfitting Gap
ax2 = axes[0, 1]
gaps = [all_results[d]['val_error_mean'] - all_results[d]['train_error_mean']
        for d in degrees_to_test]
colors = ['green' if g < 1.5 else 'orange' if g < 3.0 else 'red' for g in gaps]
ax2.bar(degrees_to_test, gaps, color=colors, alpha=0.7)
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('Overfitting Gap (%)\n(Val Error - Train Error)')
ax2.set_title('Overfitting Analysis\n(Green: minimal, Orange: moderate, Red: significant)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(degrees_to_test)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 3: Box plots of validation errors
ax3 = axes[1, 0]
val_errors_list = [all_results[d]['val_errors'] for d in degrees_to_test]
bp = ax3.boxplot(val_errors_list, labels=degrees_to_test, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_xlabel('Polynomial Degree')
ax3.set_ylabel('Validation Error (%)')
ax3.set_title('Distribution of Validation Errors\nAcross {} Repetitions'.format(n_repetitions))
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Model Complexity vs Performance
ax4 = axes[1, 1]
# Number of parameters for each degree
num_features = []
for d in degrees_to_test:
    # For degree d: sum from i=1 to d of (i+1) terms, plus 1 for bias
    n_features = sum(i+1 for i in range(1, d+1)) + 1
    num_features.append(n_features)

ax4.scatter(num_features, val_means, s=150, c=degrees_to_test,
            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
for i, d in enumerate(degrees_to_test):
    ax4.annotate(f'deg={d}', (num_features[i], val_means[i]),
                fontsize=10, ha='right', va='bottom')
ax4.set_xlabel('Number of Model Parameters')
ax4.set_ylabel('Validation Error (%)')
ax4.set_title('Model Complexity vs Performance')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_degree_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'polynomial_degree_comparison.png'")
plt.show()

print("\n" + "="*70)
print("TASK 5 COMPLETE")
print("="*70)
print(f"""
Summary:
- Best model for predictions: Degree {best_val_degree}
  (Validation Error: {all_results[best_val_degree]['val_error_mean']:.2f}%)
- Linear model (Degree 1) performance:
  {all_results[1]['val_error_mean']:.2f}% - demonstrates need for non-linearity
- Overfitting is {'observed' if any(all_results[d]['val_error_mean'] - all_results[d]['train_error_mean'] > 2.0 for d in degrees_to_test) else 'minimal'}
  in higher degree models

These results clearly show the bias-variance trade-off in model selection.
""")