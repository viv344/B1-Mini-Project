###############################################
# Runtime vs Quality Trade-off Analysis
# Section 3.2, Task 3
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt
import time
from main_project_skeleton import grad_descent, log_regr, classif_error, mean_logloss, create_features_for_poly

print("="*70)
print("ANALYSIS: n_iters vs λ - Runtime and Quality Trade-offs")
print("="*70)

# Polynomial degree
degree_poly = 3

# Configurations to test
configs = [
    # (n_iters, λ, description)
    (100, 0.01, "Slow & Stable: Small λ, Few iters"),
    (1000, 0.01, "Slow & Stable: Small λ, More iters"),
    (100, 0.1, "Balanced: Medium λ, Few iters"),
    (1000, 0.1, "Balanced: Medium λ, Moderate iters"),
    (10000, 0.1, "Balanced: Medium λ, Many iters"),
    (100, 1.0, "Fast & Aggressive: Large λ, Few iters"),
    (1000, 1.0, "Fast & Aggressive: Large λ, Moderate iters"),
]

# Create training data
n_samples_train = 400
[X_train, class_labels_train] = create_data(n_samples_train)
y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
X_train_poly = create_features_for_poly(X_train, degree_poly)
X_train_poly = np.concatenate((X_train_poly, np.ones(shape=(n_samples_train, 1))), axis=1)

# Create validation data
n_samples_val = 4000
[X_val, class_labels_val] = create_data(n_samples_val)
y_val = (class_labels_val == 1) * 0 + (class_labels_val == 2) * 1
X_val_poly = create_features_for_poly(X_val, degree_poly)
X_val_poly = np.concatenate((X_val_poly, np.ones(shape=(n_samples_val, 1))), axis=1)

# Store results
results = []

print("\nTesting different configurations...\n")
print(f"{'Config':<40} {'Runtime (s)':<15} {'Val Error (%)':<15} {'Val LogLoss':<15}")
print("-" * 85)

for n_iters, lr, description in configs:
    # Measure runtime
    start_time = time.time()
    theta_opt = grad_descent(X_train_poly, y_train, lr, n_iters)
    runtime = time.time() - start_time

    # Evaluate quality
    y_val_pred = log_regr(X_val_poly, theta_opt)
    val_error = classif_error(y_val, y_val_pred)
    val_logloss = mean_logloss(X_val_poly, y_val, theta_opt)

    results.append({
        'n_iters': n_iters,
        'lr': lr,
        'description': description,
        'runtime': runtime,
        'val_error': val_error,
        'val_logloss': val_logloss
    })

    print(f"{description:<40} {runtime:<15.4f} {val_error:<15.2f} {val_logloss:<15.4f}")

print("\n" + "="*70)
print("ANALYSIS AND INSIGHTS")
print("="*70)

# Find best quality
best_quality = min(results, key=lambda x: x['val_error'])
print(f"\nBest Quality:")
print(f"  Config: {best_quality['description']}")
print(f"  λ={best_quality['lr']}, n_iters={best_quality['n_iters']}")
print(f"  Val Error: {best_quality['val_error']:.2f}%")
print(f"  Runtime: {best_quality['runtime']:.4f}s")

# Find fastest
fastest = min(results, key=lambda x: x['runtime'])
print(f"\nFastest (but with what quality?):")
print(f"  Config: {fastest['description']}")
print(f"  λ={fastest['lr']}, n_iters={fastest['n_iters']}")
print(f"  Val Error: {fastest['val_error']:.2f}%")
print(f"  Runtime: {fastest['runtime']:.4f}s")

# Best trade-off (good quality, reasonable runtime)
# Find configs with < 5% error and sort by runtime
good_quality_configs = [r for r in results if r['val_error'] < 5.0]
if good_quality_configs:
    best_tradeoff = min(good_quality_configs, key=lambda x: x['runtime'])
    print(f"\nBest Trade-off (< 5% error, fastest):")
    print(f"  Config: {best_tradeoff['description']}")
    print(f"  λ={best_tradeoff['lr']}, n_iters={best_tradeoff['n_iters']}")
    print(f"  Val Error: {best_tradeoff['val_error']:.2f}%")
    print(f"  Runtime: {best_tradeoff['runtime']:.4f}s")

print("\n" + "="*70)
print("KEY OBSERVATIONS FOR REPORT")
print("="*70)
print("""
1. RELATIONSHIP BETWEEN n_iters AND λ:
   - They have an INVERSE relationship for achieving similar quality
   - Large λ requires fewer iterations (faster convergence per iteration)
   - Small λ requires more iterations (slower but more stable)
   - Runtime = n_iters × (time per iteration) → directly proportional to n_iters

2. EFFECT ON SOLUTION QUALITY:
   - Insufficient iterations → poor solution (high error)
   - Too large λ with few iterations → may overshoot, unstable
   - Too small λ with few iterations → insufficient progress
   - Best quality: adequate n_iters with appropriate λ

3. EFFECT ON RUNTIME:
   - Runtime scales LINEARLY with n_iters
   - Each iteration has roughly constant computational cost
   - λ does NOT affect runtime per iteration
   - Trade-off: quality vs computational time

4. REAL-WORLD PREFERENCE:
   - For PRODUCTION systems: Medium λ (0.1-0.5) with moderate n_iters (1000-5000)
     → Balances quality, speed, and stability
   - For RESEARCH/offline training: Can afford large n_iters (10000+)
     → Prioritize best quality
   - For REAL-TIME systems: Larger λ with fewer iters
     → Prioritize speed, accept slightly worse quality
   - Generally prefer: MODERATE λ with SUFFICIENT n_iters
     → More stable and predictable than aggressive settings
""")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Runtime vs n_iters for different λ
ax1 = axes[0, 0]
for lr in [0.01, 0.1, 1.0]:
    lr_results = [r for r in results if r['lr'] == lr]
    n_iters_vals = [r['n_iters'] for r in lr_results]
    runtimes = [r['runtime'] for r in lr_results]
    ax1.plot(n_iters_vals, runtimes, marker='o', label=f'λ={lr}', linewidth=2)
ax1.set_xlabel('Number of Iterations (n_iters)')
ax1.set_ylabel('Runtime (seconds)')
ax1.set_title('Runtime vs n_iters (Linear Relationship)')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Quality (validation error) vs n_iters
ax2 = axes[0, 1]
for lr in [0.01, 0.1, 1.0]:
    lr_results = [r for r in results if r['lr'] == lr]
    n_iters_vals = [r['n_iters'] for r in lr_results]
    val_errors = [r['val_error'] for r in lr_results]
    ax2.plot(n_iters_vals, val_errors, marker='o', label=f'λ={lr}', linewidth=2)
ax2.set_xlabel('Number of Iterations (n_iters)')
ax2.set_ylabel('Validation Error (%)')
ax2.set_title('Quality vs n_iters')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Quality vs Runtime (Pareto frontier)
ax3 = axes[1, 0]
runtimes = [r['runtime'] for r in results]
val_errors = [r['val_error'] for r in results]
colors = [r['lr'] for r in results]
scatter = ax3.scatter(runtimes, val_errors, c=colors, s=100, cmap='viridis', alpha=0.7)
for r in results:
    ax3.annotate(f"n={r['n_iters']}",
                 (r['runtime'], r['val_error']),
                 fontsize=8, alpha=0.7)
ax3.set_xlabel('Runtime (seconds)')
ax3.set_ylabel('Validation Error (%)')
ax3.set_title('Quality-Speed Trade-off')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Learning Rate (λ)')
ax3.grid(True, alpha=0.3)

# Plot 4: Bar chart comparing configurations
ax4 = axes[1, 1]
config_labels = [f"λ={r['lr']}\nn={r['n_iters']}" for r in results]
val_errors = [r['val_error'] for r in results]
colors_bar = ['red' if e > 5 else 'green' for e in val_errors]
ax4.barh(config_labels, val_errors, color=colors_bar, alpha=0.7)
ax4.set_xlabel('Validation Error (%)')
ax4.set_title('Quality Comparison\n(Green: < 5% error, Red: > 5%)')
ax4.axvline(x=5.0, color='black', linestyle='--', linewidth=1, label='5% threshold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('runtime_quality_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'runtime_quality_analysis.png'")
plt.show()

print("\n" + "="*70)
print("RECOMMENDATION FOR REPORT")
print("="*70)
print("""
For a real-world application, I would prefer:
  → MODERATE n_iters (1000-5000) with MODERATE λ (0.1-0.5)

Reasoning:
1. Provides good quality (< 5% error) without excessive computation
2. More stable and robust than aggressive large λ settings
3. Fast enough for practical deployment (< 1 second training)
4. Can be fine-tuned based on specific application requirements:
   - Time-critical: Reduce n_iters, increase λ slightly
   - Quality-critical: Increase n_iters, keep moderate λ
5. Avoids extremes that may cause issues in production
""")
