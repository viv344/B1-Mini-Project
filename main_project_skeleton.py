###############################################
# Author & Copyright: Konstantinos Kamnitsas
# B1 - Project - 2025
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt  # For plotting


def grad_descent(X_train, y_train, learning_rate, iters_total, track_loss=False):
    # X_train: Matrix of dimensions: number_of_samples x (number_of_features+1)
    # y_train: Vector of dimensions: number_of_samples
    # track_loss: If True, returns loss history for plotting convergence
    # Returns: Optimized theta, vector of dimensions: (number_of_features+1)
    #          If track_loss=True, also returns list of losses at each iteration

    # Initialize Theta
    theta = np.zeros(shape=(X_train.shape[1], 1))

    # Reshape y_train to column vector if needed
    y_train = y_train.reshape(-1, 1)

    # Number of samples
    n = X_train.shape[0]

    # Track loss history if requested
    loss_history = [] if track_loss else None

    # Train Theta with Gradient Descent
    for iteration in range(iters_total):
        # Compute predictions: y_pred = σ(X̂θ)
        # σ(z) = 1 / (1 + e^(-z))
        z = X_train @ theta  # Matrix multiplication: X̂θ
        y_pred = 1 / (1 + np.exp(-z))  # Sigmoid function

        # Track loss if requested
        if track_loss:
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -y_train * np.log(y_pred_clipped) - (1 - y_train) * np.log(1 - y_pred_clipped)
            mean_loss = np.mean(loss)
            loss_history.append(mean_loss)

        # Compute gradient: ∇θL_C(θ, x̂, y) = (1/n) * X̂^T * (ȳ - y)
        # From Eq. 7 in PDF: ∇θL_C(θ, x̂, y) = (ȳ - y)x̂
        gradient = (1/n) * (X_train.T @ (y_pred - y_train))

        # Update parameters: θ = θ - λ * ∇θJ
        theta = theta - learning_rate * gradient

    theta_opt = theta

    if track_loss:
        return theta_opt, loss_history
    else:
        return theta_opt


def  mean_logloss(X, y_real, theta):
    """
    Calculate Mean Log-Loss (cost function) for logistic regression.
    X: Matrix of dimensions: number_of_samples x (number_of_features+1)
    y_real: True labels, vector of dimensions: number_of_samples
    theta: Model parameters, vector of dimensions: (number_of_features+1) x 1
    Returns: Mean Log-Loss (scalar)
    """
    # Get predictions using logistic regression
    y_pred = log_regr(X, theta)

    # Reshape y_real to column vector if needed
    y_real = y_real.reshape(-1, 1)

    # Number of samples
    n = X.shape[0]

    # Calculate Log-Loss for each sample: L_C(θ, x̂, y) = -y*log(ȳ) - (1-y)*log(1-ȳ)
    # From Eq. 3a in the PDF
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -y_real * np.log(y_pred) - (1 - y_real) * np.log(1 - y_pred)

    # Mean Log-Loss (cost function): J_C(θ, X̂, y) = (1/n) * Σ L_C(θ, x̂_i, y_i)
    # From Eq. 4 in the PDF
    mean_logloss = np.mean(loss)

    return mean_logloss



def log_regr(X, theta):
    """
    Logistic Regression prediction function.
    X: Matrix of dimensions: number_of_samples x (number_of_features+1)
    theta: Model parameters, vector of dimensions: (number_of_features+1) x 1
    Returns: Predicted probabilities ȳ ∈ [0,1], vector of dimensions: number_of_samples
    """
    # Compute ȳ = σ(X̂θ) = 1 / (1 + e^(-X̂θ))
    # This is Eq. 1b and Eq. 2 from the PDF
    z = X @ theta  # Matrix multiplication: X̂θ
    y_pred = 1 / (1 + np.exp(-z))  # Sigmoid function

    return y_pred


def classif_error(y_real, y_pred):
    """
    Calculate classification error percentage.
    y_real: True labels (can be {0,1} or {1,2})
    y_pred: Predicted probabilities ȳ ∈ [0,1] OR predicted class labels {0,1} or {1,2}
    Returns: Error percentage (%)
    """
    # Convert y_pred to class labels if they are probabilities
    if np.max(y_pred) <= 1.0 and np.min(y_pred) >= 0.0 and not np.all(np.isin(y_pred, [0, 1])):
        # y_pred contains probabilities, convert to binary labels
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    else:
        # y_pred already contains class labels
        y_pred_labels = y_pred.flatten()

    y_real_flat = y_real.flatten()

    # If y_real contains {1,2}, convert both to {0,1}
    if np.max(y_real_flat) == 2:
        y_real_flat = (y_real_flat == 2).astype(int)
    if np.max(y_pred_labels) == 2:
        y_pred_labels = (y_pred_labels == 2).astype(int)

    # Calculate error: e = 100 × (number of wrong predictions / total predictions)
    # From Eq. 8 in the PDF
    total_predictions = len(y_real_flat)
    wrong_predictions = np.sum(y_real_flat != y_pred_labels)
    err_perc = 100 * (wrong_predictions / total_predictions)

    return err_perc


def create_features_for_poly(X, degree):
    """
    Create polynomial features up to specified degree for 2D input.
    X: Matrix of dimensions number_of_samples x 2 (original features x1, x2)
    degree: Degree of polynomial
    Returns: Matrix with polynomial features (does NOT include bias term of 1)

    Example for degree=2: Returns [x1^2, x1*x2, x2^2, x1, x2]
    Example for degree=3: Returns [x1^3, x1^2*x2, x1*x2^2, x2^3, x1^2, x1*x2, x2^2, x1, x2]
    """
    n_samples = X.shape[0]
    x1 = X[:, 0].reshape(-1, 1)  # First feature as column vector
    x2 = X[:, 1].reshape(-1, 1)  # Second feature as column vector

    # List to store all polynomial features
    all_features = []

    # Build features from highest degree down to degree 1
    # This follows Eq. 9 from the PDF
    for d in range(degree, 0, -1):
        # For each degree d, create all terms where powers sum to d
        # Terms: x1^d, x1^(d-1)*x2, ..., x1*x2^(d-1), x2^d
        for i in range(d + 1):
            power_x1 = d - i
            power_x2 = i
            # Create term: x1^(d-i) * x2^i
            term = (x1 ** power_x1) * (x2 ** power_x2)
            all_features.append(term)

    # Concatenate all features horizontally
    features_poly = np.concatenate(all_features, axis=1)

    return features_poly



# --------- Helper Plotting Function ----------------

def plot_data(x, class_labels):
    """
     Plots the data returned from the create_data() function.
     x: Matrix of dimensions number_of_samples x number_of_features.
        This should NOT include the concatenated 1 for the bias.
     class_labels: Vector of dimensions number_of_samples.
                   Expects values class_labels={1,2} . Not the y={0,1}
    """
    # Plot the points
    size_markers = 20

    fig, ax = plt.subplots()
    # Class-1 is Red.
    ax.scatter(x[class_labels==1, 0], x[class_labels==1, 1], s=size_markers, c='red', edgecolors='black', linewidth=1.0)
    # Class-2 is Green
    ax.scatter(x[class_labels==2, 0], x[class_labels==2, 1], s=size_markers, c='green', edgecolors='black', linewidth=1.0)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim([-2.0, 3.0]);
    ax.set_ylim([-2.0, 3.0]);
    ax.legend('class 1', 'class 2');
    ax.grid(True)

    plt.show()




####################### THE MAIN FUNCTION #####################################

if __name__ == "__main__":
    """
     NOTE: Below is a skeleton code to get you started.
     If you simply run the code, without any changes, it should run,
     sample some training and validation data, and plot them.
     (It may need you to install Numpy and Matplotlib libraries if they are not installed already though!)

     You can build on this skeleton for coding the project.
     However you do NOT need to follow this structure. If you prefer,
     follow your own workflow and change the code as much as you like.
     The only code that you MUST NOT MODIFY is the data-creating
     function in create_data.m
    """

    # Hyper-parameters:
    lr = 0.0001  # learning rate
    gd_iters = 100
    degree_poly = 1

    # Create training data
    n_samples_train = 400
    [X_train, class_labels_train] = create_data(n_samples_train)
    # Change class labels ={1,2} to values for y={0,1} respectively.
    y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
    # Make poly
    X_train_poly = create_features_for_poly(X_train, degree_poly)
    # concat 1 for bias
    X_train_poly = np.concatenate((X_train_poly, np.ones(shape=(n_samples_train, 1))), axis=1)


    # Create validation data
    n_samples_val = 4000
    [X_val, class_labels_val] = create_data(n_samples_val)
    # Change class labels ={1,2} to values for y={0,1} respectively.
    y_val = (class_labels_val==1) * 0 + (class_labels_val==2) * 1
    # Make poly
    X_val_poly = create_features_for_poly(X_val, degree_poly)
    # concat 1 for bias
    X_val_poly = np.concatenate((X_val_poly, np.ones(shape=(n_samples_val, 1))), axis=1)


    # Optimize - Logistic Regression - Gradient Descent
    print("\n" + "="*50)
    print("Training model with Gradient Descent...")
    print(f"Learning rate: {lr}, Iterations: {gd_iters}, Polynomial degree: {degree_poly}")
    print("="*50)

    theta_opt = grad_descent(X_train_poly, y_train, lr, gd_iters)

    # Evaluate performance:
    print("\nOptimized theta parameters:")
    print(theta_opt.flatten())

    # Training performance
    y_train_pred = log_regr(X_train_poly, theta_opt)
    train_error = classif_error(y_train, y_train_pred)
    train_logloss = mean_logloss(X_train_poly, y_train, theta_opt)

    # Validation performance
    y_val_pred = log_regr(X_val_poly, theta_opt)
    val_error = classif_error(y_val, y_val_pred)
    val_logloss = mean_logloss(X_val_poly, y_val, theta_opt)

    print("\n" + "="*50)
    print("PERFORMANCE RESULTS")
    print("="*50)
    print(f"Training Error:     {train_error:.2f}%")
    print(f"Training Log-Loss:  {train_logloss:.4f}")
    print(f"\nValidation Error:   {val_error:.2f}%")
    print(f"Validation Log-Loss: {val_logloss:.4f}")
    print("="*50 + "\n")

    # Plot data:
    plot_data(X_train, class_labels_train)
    plot_data(X_val, class_labels_val)
