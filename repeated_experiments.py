###############################################
# Repeated Experiments - Average Away Randomness
# Section 3.2, Task 4
###############################################

from create_data import create_data
import numpy as np
from main_project_skeleton import grad_descent, log_regr, classif_error, mean_logloss, create_features_for_poly

def run_experiment(degree_poly, lr, n_iters, n_samples_train=400, n_samples_val=4000):
    """
    Run a single training-validation experiment.
    Returns: (train_error, val_error, train_logloss, val_logloss)
    """
    # Create training data
    [X_train, class_labels_train] = create_data(n_samples_train)
    y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
    X_train_poly = create_features_for_poly(X_train, degree_poly)
    X_train_poly = np.concatenate((X_train_poly, np.ones(shape=(n_samples_train, 1))), axis=1)

    # Create validation data
    [X_val, class_labels_val] = create_data(n_samples_val)
    y_val = (class_labels_val == 1) * 0 + (class_labels_val == 2) * 1
    X_val_poly = create_features_for_poly(X_val, degree_poly)
    X_val_poly = np.concatenate((X_val_poly, np.ones(shape=(n_samples_val, 1))), axis=1)

    # Train model
    theta_opt = grad_descent(X_train_poly, y_train, lr, n_iters)

    # Evaluate on training data
    y_train_pred = log_regr(X_train_poly, theta_opt)
    train_error = classif_error(y_train, y_train_pred)
    train_logloss = mean_logloss(X_train_poly, y_train, theta_opt)

    # Evaluate on validation data
    y_val_pred = log_regr(X_val_poly, theta_opt)
    val_error = classif_error(y_val, y_val_pred)
    val_logloss = mean_logloss(X_val_poly, y_val, theta_opt)

    return train_error, val_error, train_logloss, val_logloss


def repeat_experiments(degree_poly, lr, n_iters, n_repetitions=20, n_samples_train=400, n_samples_val=4000):
    """
    Repeat the training-validation experiment multiple times.
    Returns: dict with mean and std for train/val errors and logloss
    """
    train_errors = []
    val_errors = []
    train_loglosses = []
    val_loglosses = []

    print(f"Running {n_repetitions} repetitions...")
    for i in range(n_repetitions):
        train_err, val_err, train_ll, val_ll = run_experiment(
            degree_poly, lr, n_iters, n_samples_train, n_samples_val
        )
        train_errors.append(train_err)
        val_errors.append(val_err)
        train_loglosses.append(train_ll)
        val_loglosses.append(val_ll)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_repetitions} repetitions")

    results = {
        'train_error_mean': np.mean(train_errors),
        'train_error_std': np.std(train_errors),
        'val_error_mean': np.mean(val_errors),
        'val_error_std': np.std(val_errors),
        'train_logloss_mean': np.mean(train_loglosses),
        'train_logloss_std': np.std(train_loglosses),
        'val_logloss_mean': np.mean(val_loglosses),
        'val_logloss_std': np.std(val_loglosses),
        'train_errors': train_errors,
        'val_errors': val_errors,
    }

    return results


if __name__ == "__main__":
    print("="*70)
    print("REPEATED EXPERIMENTS - Averaging Away Data Sampling Randomness")
    print("="*70)

    # Use best hyper-parameters from Task 1
    best_lr = 0.1
    best_n_iters = 10000
    degree_poly = 3
    n_repetitions = 20

    print(f"\nConfiguration:")
    print(f"  Polynomial degree: {degree_poly}")
    print(f"  Learning rate (λ): {best_lr}")
    print(f"  Iterations: {best_n_iters}")
    print(f"  Number of repetitions: {n_repetitions}")
    print(f"  Training samples per repetition: 400")
    print(f"  Validation samples per repetition: 4000")
    print()

    # Run repeated experiments
    results = repeat_experiments(degree_poly, best_lr, best_n_iters, n_repetitions)

    # Display results
    print("\n" + "="*70)
    print("RESULTS - Averaged Over {} Repetitions".format(n_repetitions))
    print("="*70)

    print(f"\nTraining Performance:")
    print(f"  Error:    {results['train_error_mean']:.2f}% ± {results['train_error_std']:.2f}%")
    print(f"  Log-Loss: {results['train_logloss_mean']:.4f} ± {results['train_logloss_std']:.4f}")

    print(f"\nValidation Performance:")
    print(f"  Error:    {results['val_error_mean']:.2f}% ± {results['val_error_std']:.2f}%")
    print(f"  Log-Loss: {results['val_logloss_mean']:.4f} ± {results['val_logloss_std']:.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
The ± values show the standard deviation across repetitions, indicating:
- How much performance varies due to random data sampling
- Smaller std = more stable/reliable performance estimate
- The mean values are more representative of true model performance
  than any single experiment result

These averaged results should be used when comparing different models
or configurations, as they account for randomness in data generation.
""")

    # Show individual repetition results
    print("="*70)
    print("INDIVIDUAL REPETITION RESULTS")
    print("="*70)
    print(f"{'Rep':<5} {'Train Err (%)':<15} {'Val Err (%)':<15}")
    print("-" * 35)
    for i, (train_err, val_err) in enumerate(zip(results['train_errors'], results['val_errors'])):
        print(f"{i+1:<5} {train_err:<15.2f} {val_err:<15.2f}")

    print("\n" + "="*70)
    print("VARIABILITY ANALYSIS")
    print("="*70)
    print(f"Training Error Range: [{min(results['train_errors']):.2f}%, {max(results['train_errors']):.2f}%]")
    print(f"Validation Error Range: [{min(results['val_errors']):.2f}%, {max(results['val_errors']):.2f}%]")
    print(f"Validation Error Coefficient of Variation: {100 * results['val_error_std'] / results['val_error_mean']:.2f}%")

    print("\n" + "="*70)
    print("TASK 4 COMPLETE")
    print("="*70)
    print("""
This infrastructure will be used for:
- Task 5: Comparing polynomial degrees (1, 2, 3, 4, 5)
- Task 6: Comparing training set sizes (50, 100, 200, 400)

Each comparison will use repeated experiments to get reliable estimates.
""")
