###############################################
# Hyper-parameter Tuning Experiment
# Section 3.2, Task 1
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt
from main_project_skeleton import grad_descent, log_regr, classif_error, mean_logloss, create_features_for_poly

# Hyper-parameter values to test
n_iters_values = [100, 500, 1000, 10000]
learning_rates = [0.01, 0.1, 0.5, 1.0]

# Use polynomial degree 3 for finding best hyper-parameters (as recommended in PDF)
degree_poly = 3

print("="*70)
print("HYPER-PARAMETER TUNING EXPERIMENT")
print("="*70)
print(f"Testing with polynomial degree: {degree_poly}")
print(f"Learning rates: {learning_rates}")
print(f"Iterations: {n_iters_values}")
print("="*70)

# Create training data (400 samples)
n_samples_train = 400
[X_train, class_labels_train] = create_data(n_samples_train)
y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
X_train_poly = create_features_for_poly(X_train, degree_poly)
X_train_poly = np.concatenate((X_train_poly, np.ones(shape=(n_samples_train, 1))), axis=1)

# Create validation data (4000 samples)
n_samples_val = 4000
[X_val, class_labels_val] = create_data(n_samples_val)
y_val = (class_labels_val == 1) * 0 + (class_labels_val == 2) * 1
X_val_poly = create_features_for_poly(X_val, degree_poly)
X_val_poly = np.concatenate((X_val_poly, np.ones(shape=(n_samples_val, 1))), axis=1)

# Store results
results = []

# Grid search over hyper-parameters
for lr in learning_rates:
    for n_iters in n_iters_values:
        print(f"\nTraining with lr={lr}, n_iters={n_iters}...", end=" ")

        # Train model
        theta_opt = grad_descent(X_train_poly, y_train, lr, n_iters)

        # Evaluate on validation data
        y_val_pred = log_regr(X_val_poly, theta_opt)
        val_error = classif_error(y_val, y_val_pred)
        val_logloss = mean_logloss(X_val_poly, y_val, theta_opt)

        # Store results
        results.append({
            'lr': lr,
            'n_iters': n_iters,
            'val_error': val_error,
            'val_logloss': val_logloss
        })

        print(f"Val Error: {val_error:.2f}%, Val LogLoss: {val_logloss:.4f}")

print("\n" + "="*70)
print("RESULTS TABLE - Validation Error (%)")
print("="*70)

# Print table header
print(f"{'λ (lr)':<10}", end="")
for n_iters in n_iters_values:
    print(f"n_iters={n_iters:<6}", end="  ")
print()
print("-" * 70)

# Print table rows
for lr in learning_rates:
    print(f"{lr:<10.2f}", end="")
    for n_iters in n_iters_values:
        # Find result for this combination
        result = [r for r in results if r['lr'] == lr and r['n_iters'] == n_iters][0]
        print(f"{result['val_error']:>14.2f}%", end="  ")
    print()

print("\n" + "="*70)
print("RESULTS TABLE - Validation Log-Loss")
print("="*70)

# Print table header
print(f"{'λ (lr)':<10}", end="")
for n_iters in n_iters_values:
    print(f"n_iters={n_iters:<6}", end="  ")
print()
print("-" * 70)

# Print table rows
for lr in learning_rates:
    print(f"{lr:<10.2f}", end="")
    for n_iters in n_iters_values:
        # Find result for this combination
        result = [r for r in results if r['lr'] == lr and r['n_iters'] == n_iters][0]
        print(f"{result['val_logloss']:>14.4f}", end="  ")
    print()

# Find best hyper-parameters
best_result = min(results, key=lambda x: x['val_error'])

print("\n" + "="*70)
print("BEST HYPER-PARAMETERS (Based on Validation Error)")
print("="*70)
print(f"Learning rate (λ): {best_result['lr']}")
print(f"Iterations (n_iters): {best_result['n_iters']}")
print(f"Validation Error: {best_result['val_error']:.2f}%")
print(f"Validation Log-Loss: {best_result['val_logloss']:.4f}")
print("="*70)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Validation Error
for lr in learning_rates:
    errors = [r['val_error'] for r in results if r['lr'] == lr]
    ax1.plot(n_iters_values, errors, marker='o', label=f'λ={lr}')

ax1.set_xlabel('Number of Iterations')
ax1.set_ylabel('Validation Error (%)')
ax1.set_title('Validation Error vs Iterations for Different Learning Rates')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True)

# Plot 2: Validation Log-Loss
for lr in learning_rates:
    losses = [r['val_logloss'] for r in results if r['lr'] == lr]
    ax2.plot(n_iters_values, losses, marker='o', label=f'λ={lr}')

ax2.set_xlabel('Number of Iterations')
ax2.set_ylabel('Validation Log-Loss')
ax2.set_title('Validation Log-Loss vs Iterations for Different Learning Rates')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_results.png', dpi=150)
print("\nPlot saved as 'hyperparameter_tuning_results.png'")
plt.show()
