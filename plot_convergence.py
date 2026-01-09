###############################################
# Loss vs Iterations - Convergence Analysis
# Section 3.2, Task 2
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt
from main_project_skeleton import grad_descent, create_features_for_poly

print("="*70)
print("CONVERGENCE ANALYSIS - Loss vs Iterations")
print("="*70)

# Use the best hyper-parameters found from Task 1
best_lr = 0.1
best_n_iters = 10000
degree_poly = 3

# Create training data (400 samples)
n_samples_train = 400
[X_train, class_labels_train] = create_data(n_samples_train)
y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
X_train_poly = create_features_for_poly(X_train, degree_poly)
X_train_poly = np.concatenate((X_train_poly, np.ones(shape=(n_samples_train, 1))), axis=1)

print(f"\nTraining with best hyper-parameters:")
print(f"Learning rate (λ): {best_lr}")
print(f"Iterations: {best_n_iters}")
print(f"Polynomial degree: {degree_poly}")
print("\nTraining and tracking loss...")

# Train with loss tracking
theta_opt, loss_history = grad_descent(X_train_poly, y_train, best_lr, best_n_iters, track_loss=True)

print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Loss reduction: {loss_history[0] - loss_history[-1]:.4f}")
print(f"Loss reduction (%): {100 * (loss_history[0] - loss_history[-1]) / loss_history[0]:.2f}%")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full training curve (linear scale)
ax1 = axes[0, 0]
ax1.plot(range(len(loss_history)), loss_history, linewidth=1.5)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Mean Log-Loss')
ax1.set_title(f'Convergence: Loss vs Iterations\n(λ={best_lr}, n_iters={best_n_iters}, degree={degree_poly})')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, best_n_iters])

# Plot 2: Log scale for y-axis to see convergence better
ax2 = axes[0, 1]
ax2.plot(range(len(loss_history)), loss_history, linewidth=1.5, color='orange')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Mean Log-Loss (log scale)')
ax2.set_title('Convergence (Log Scale)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim([0, best_n_iters])

# Plot 3: First 1000 iterations (zoomed in)
ax3 = axes[1, 0]
ax3.plot(range(min(1000, len(loss_history))), loss_history[:1000], linewidth=1.5, color='green')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Mean Log-Loss')
ax3.set_title('First 1000 Iterations (Zoomed In)')
ax3.grid(True, alpha=0.3)

# Plot 4: Last 1000 iterations (to check stabilization)
ax4 = axes[1, 1]
start_idx = max(0, len(loss_history) - 1000)
ax4.plot(range(start_idx, len(loss_history)), loss_history[start_idx:], linewidth=1.5, color='red')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Mean Log-Loss')
ax4.set_title('Last 1000 Iterations (Check Stabilization)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'convergence_analysis.png'")
plt.show()

# Additional analysis: Compare different learning rates
print("\n" + "="*70)
print("COMPARING CONVERGENCE FOR DIFFERENT LEARNING RATES")
print("="*70)

learning_rates_to_compare = [0.01, 0.1, 0.5, 1.0]
n_iters_compare = 2000  # Use fewer iterations for comparison

fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

for lr in learning_rates_to_compare:
    print(f"\nTraining with λ={lr}...")
    theta, loss_hist = grad_descent(X_train_poly, y_train, lr, n_iters_compare, track_loss=True)

    # Linear scale
    ax_left.plot(range(len(loss_hist)), loss_hist, label=f'λ={lr}', linewidth=1.5)

    # Log scale
    ax_right.plot(range(len(loss_hist)), loss_hist, label=f'λ={lr}', linewidth=1.5)

    print(f"  Initial loss: {loss_hist[0]:.4f}, Final loss: {loss_hist[-1]:.4f}")

ax_left.set_xlabel('Iteration')
ax_left.set_ylabel('Mean Log-Loss')
ax_left.set_title('Convergence Comparison: Different Learning Rates')
ax_left.legend()
ax_left.grid(True, alpha=0.3)

ax_right.set_xlabel('Iteration')
ax_right.set_ylabel('Mean Log-Loss (log scale)')
ax_right.set_title('Convergence Comparison (Log Scale)')
ax_right.set_yscale('log')
ax_right.legend()
ax_right.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('convergence_comparison_learning_rates.png', dpi=150, bbox_inches='tight')
print("\nComparison plot saved as 'convergence_comparison_learning_rates.png'")
plt.show()

print("\n" + "="*70)
print("CONVERGENCE ANALYSIS COMPLETE")
print("="*70)
print("\nKey Observations:")
print("- Check if loss has stabilized by the end of training")
print("- Larger learning rates converge faster but may oscillate")
print("- Smaller learning rates are more stable but slower")
print("- The chosen λ=0.1 provides good balance between speed and stability")
