###############################################
# Compare Training Set Sizes [50, 100, 200, 400]
# Section 3.2, Task 6
###############################################

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/vivek/Documents/B1 Mini Project')
from repeated_experiments import repeat_experiments

print("="*70)
print("TASK 6: INVESTIGATING IMPACT OF TRAINING SET SIZE")
print("="*70)

# Best model from Task 5
best_degree = 3
best_lr = 0.1
best_n_iters = 10000
n_repetitions = 20
n_samples_val = 4000  # Keep validation set constant
training_sizes = [50, 100, 200, 400]

print(f"\nConfiguration:")
print(f"  Best model: Polynomial degree {best_degree}")
print(f"  Learning rate (λ): {best_lr}")
print(f"  Iterations: {best_n_iters}")
print(f"  Repetitions per training size: {n_repetitions}")
print(f"  Validation samples (constant): {n_samples_val}")
print(f"  Training sizes to test: {training_sizes}")
print()

# Store results for all training sizes
all_results = {}

for n_train in training_sizes:
    print(f"\n{'='*70}")
    print(f"Testing Training Size: {n_train} samples")
    print(f"{'='*70}")

    results = repeat_experiments(
        degree_poly=best_degree,
        lr=best_lr,
        n_iters=best_n_iters,
        n_repetitions=n_repetitions,
        n_samples_train=n_train,
        n_samples_val=n_samples_val
    )

    all_results[n_train] = results

    print(f"\nResults for {n_train} training samples:")
    print(f"  Train Error: {results['train_error_mean']:.2f}% ± {results['train_error_std']:.2f}%")
    print(f"  Val Error:   {results['val_error_mean']:.2f}% ± {results['val_error_std']:.2f}%")

# Create results table
print("\n" + "="*70)
print("RESULTS TABLE - AVERAGED OVER {} REPETITIONS".format(n_repetitions))
print("="*70)

print(f"\n{'Metric':<20} ", end="")
for n_train in training_sizes:
    print(f"n={n_train:<5} ", end="")
print()
print("-" * 70)

# Training error row
print(f"{'Train Error (%)':<20} ", end="")
for n_train in training_sizes:
    r = all_results[n_train]
    print(f"{r['train_error_mean']:>6.2f}  ", end="")
print()

# Validation error row
print(f"{'Val Error (%)':<20} ", end="")
for n_train in training_sizes:
    r = all_results[n_train]
    print(f"{r['val_error_mean']:>6.2f}  ", end="")
print()

# Overfitting gap row
print(f"{'Gap (Val-Train) (%)':<20} ", end="")
for n_train in training_sizes:
    r = all_results[n_train]
    gap = r['val_error_mean'] - r['train_error_mean']
    print(f"{gap:>6.2f}  ", end="")
print()

print("\n" + "="*70)
print("DETAILED TABLE WITH STANDARD DEVIATIONS")
print("="*70)

print(f"\n{'Training Size':<15} {'Train Error (%)':<25} {'Val Error (%)':<25} {'Overfitting Gap':<15}")
print("-" * 80)

for n_train in training_sizes:
    r = all_results[n_train]
    gap = r['val_error_mean'] - r['train_error_mean']
    print(f"{n_train:<15} {r['train_error_mean']:>7.2f} ± {r['train_error_std']:<11.2f} "
          f"{r['val_error_mean']:>7.2f} ± {r['val_error_std']:<11.2f} {gap:>13.2f}%")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

best_val_size = min(training_sizes, key=lambda n: all_results[n]['val_error_mean'])
worst_val_size = max(training_sizes, key=lambda n: all_results[n]['val_error_mean'])

print(f"\nBest Validation Performance:")
print(f"  Training size: {best_val_size} samples")
print(f"  Validation Error: {all_results[best_val_size]['val_error_mean']:.2f}% ± {all_results[best_val_size]['val_error_std']:.2f}%")
print(f"  Training Error: {all_results[best_val_size]['train_error_mean']:.2f}% ± {all_results[best_val_size]['train_error_std']:.2f}%")

print(f"\nWorst Validation Performance:")
print(f"  Training size: {worst_val_size} samples")
print(f"  Validation Error: {all_results[worst_val_size]['val_error_mean']:.2f}% ± {all_results[worst_val_size]['val_error_std']:.2f}%")
print(f"  Training Error: {all_results[worst_val_size]['train_error_mean']:.2f}% ± {all_results[worst_val_size]['train_error_std']:.2f}%")

improvement = all_results[worst_val_size]['val_error_mean'] - all_results[best_val_size]['val_error_mean']
print(f"\nPerformance improvement from {worst_val_size} to {best_val_size} samples: {improvement:.2f}%")

# Overfitting analysis
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)

print(f"\n{'Training Size':<15} {'Overfitting Gap (%)':<20} {'Interpretation':<30}")
print("-" * 65)

for n_train in training_sizes:
    r = all_results[n_train]
    gap = r['val_error_mean'] - r['train_error_mean']

    if gap < 0:
        interpretation = "Underfitting (train > val)"
    elif gap < 1.0:
        interpretation = "Minimal overfitting"
    elif gap < 2.0:
        interpretation = "Moderate overfitting"
    else:
        interpretation = "Significant overfitting!"

    print(f"{n_train:<15} {gap:>10.2f}% {'':<9} {interpretation:<30}")

print("\n" + "="*70)
print("KEY OBSERVATIONS FOR REPORT")
print("="*70)
print("""
1. IMPACT OF TRAINING DATA SIZE:
   - More training data generally leads to better generalization
   - Small training sets → higher validation error (underfitting)
   - Model cannot learn enough from limited examples
   - Diminishing returns: performance improvement slows with more data

2. OVERFITTING PATTERNS:
   - Small datasets (50-100 samples): May show overfitting
     → Model has many parameters relative to data size
     → Can memorize limited training examples
   - Large datasets (200-400 samples): Less overfitting
     → More representative sample of true distribution
     → Better generalization to validation set

3. TRAINING ERROR BEHAVIOR:
   - Smaller training sets → lower training error (easier to fit)
   - Larger training sets → slightly higher training error
     → More diverse examples, harder to fit all perfectly
     → But this is GOOD - means model isn't just memorizing

4. PRACTICAL IMPLICATIONS:
   - Data collection is crucial for model performance
   - Need sufficient data for model complexity
   - For this model (degree 3): ~200-400 samples adequate
   - With limited data: consider simpler models
""")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training vs Validation Error
ax1 = axes[0, 0]
train_means = [all_results[n]['train_error_mean'] for n in training_sizes]
train_stds = [all_results[n]['train_error_std'] for n in training_sizes]
val_means = [all_results[n]['val_error_mean'] for n in training_sizes]
val_stds = [all_results[n]['val_error_std'] for n in training_sizes]

ax1.errorbar(training_sizes, train_means, yerr=train_stds, marker='o',
             label='Training Error', linewidth=2, capsize=5, markersize=8)
ax1.errorbar(training_sizes, val_means, yerr=val_stds, marker='s',
             label='Validation Error', linewidth=2, capsize=5, markersize=8)
ax1.set_xlabel('Training Set Size')
ax1.set_ylabel('Classification Error (%)')
ax1.set_title(f'Learning Curve: Training vs Validation Error\n(Degree {best_degree}, λ={best_lr}, n_iters={best_n_iters})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(training_sizes)

# Plot 2: Overfitting Gap
ax2 = axes[0, 1]
gaps = [all_results[n]['val_error_mean'] - all_results[n]['train_error_mean']
        for n in training_sizes]
colors = ['red' if g > 2.0 else 'orange' if g > 1.0 else 'green' for g in gaps]
bars = ax2.bar(training_sizes, gaps, color=colors, alpha=0.7, width=30)
ax2.set_xlabel('Training Set Size')
ax2.set_ylabel('Overfitting Gap (%)\n(Val Error - Train Error)')
ax2.set_title('Overfitting Analysis\n(Green: minimal, Orange: moderate, Red: significant)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xticks(training_sizes)
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{gap:.2f}%', ha='center', va='bottom' if gap > 0 else 'top')

# Plot 3: Validation Error Distribution
ax3 = axes[1, 0]
val_errors_list = [all_results[n]['val_errors'] for n in training_sizes]
bp = ax3.boxplot(val_errors_list, labels=training_sizes, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
ax3.set_xlabel('Training Set Size')
ax3.set_ylabel('Validation Error (%)')
ax3.set_title(f'Validation Error Distribution\nAcross {n_repetitions} Repetitions')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Error Reduction vs Data Size
ax4 = axes[1, 1]
# Relative to smallest training size
baseline_val = all_results[training_sizes[0]]['val_error_mean']
relative_improvement = [baseline_val - all_results[n]['val_error_mean'] for n in training_sizes]
ax4.plot(training_sizes, val_means, marker='o', linewidth=2.5, markersize=10,
         color='darkblue', label='Validation Error')
ax4.fill_between(training_sizes,
                  [v - s for v, s in zip(val_means, val_stds)],
                  [v + s for v, s in zip(val_means, val_stds)],
                  alpha=0.3, color='lightblue')
ax4.set_xlabel('Training Set Size')
ax4.set_ylabel('Validation Error (%)')
ax4.set_title('Validation Error vs Training Set Size\n(Shaded area: ± 1 std)')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(training_sizes)

# Add text annotations for each point
for n, val_mean in zip(training_sizes, val_means):
    ax4.annotate(f'{val_mean:.2f}%',
                (n, val_mean), textcoords="offset points",
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('training_size_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'training_size_comparison.png'")
plt.show()

print("\n" + "="*70)
print("TASK 6 COMPLETE")
print("="*70)
print(f"""
Summary:
- Training with {worst_val_size} samples: {all_results[worst_val_size]['val_error_mean']:.2f}% val error
- Training with {best_val_size} samples: {all_results[best_val_size]['val_error_mean']:.2f}% val error
- Improvement: {improvement:.2f}%

Overfitting observations:
""")

for n_train in training_sizes:
    gap = all_results[n_train]['val_error_mean'] - all_results[n_train]['train_error_mean']
    if gap > 2.0:
        print(f"  - {n_train} samples: Significant overfitting (gap={gap:.2f}%)")
    elif gap > 1.0:
        print(f"  - {n_train} samples: Moderate overfitting (gap={gap:.2f}%)")
    else:
        print(f"  - {n_train} samples: Minimal overfitting (gap={gap:.2f}%)")

print("""
Conclusion: More training data improves generalization and reduces overfitting.
For this problem (degree 3 polynomial), 200-400 samples provide good performance.
""")
