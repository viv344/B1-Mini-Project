###############################################
# Author & Copyright: Konstantinos Kamnitsas
# B1 - Project - 2025
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt  # For plotting


def grad_descent(X_train, y_train, learning_rate, iters_total):
    # X_train: Matrix of dimensions: number_of_samples x (number_of_features+1)
    # y_train: Vector of dimensions: number_of_samples
    # Returns: Optimized theta, vector of dimensions: (number_of_features+1)

    # Initialize Theta
    theta = np.zeros(shape=(X_train.shape[1], 1))

    # Reshape y_train to column vector if needed
    y_train = y_train.reshape(-1, 1)

    # Number of samples
    n = X_train.shape[0]

    # Train Theta with Gradient Descent
    for iteration in range(iters_total):
        # Compute predictions: y_pred = σ(X̂θ)
        # σ(z) = 1 / (1 + e^(-z))
        z = X_train @ theta  # Matrix multiplication: X̂θ
        y_pred = 1 / (1 + np.exp(-z))  # Sigmoid function

        # Compute gradient: ∇θL_C(θ, x̂, y) = (1/n) * X̂^T * (ȳ - y)
        # From Eq. 7 in PDF: ∇θL_C(θ, x̂, y) = (ȳ - y)x̂
        gradient = (1/n) * (X_train.T @ (y_pred - y_train))

        # Update parameters: θ = θ - λ * ∇θJ
        theta = theta - learning_rate * gradient

    theta_opt = theta

    return theta_opt


def  mean_logloss(X, y_real, theta):
    # Implement ...
    mean_logloss = 999  # placeholder

    return mean_logloss



def log_regr(X, theta):
    # Implement ...
    y_pred = 999  # placeholder
    return y_pred


def classif_error(y_real, y_pred):
    # Implement...
    err_perc = 999  # placeholder
    return err_perc


def create_features_for_poly(X, degree):
    # Implement ...
    #...
    features_poly = X  # Replace. Placeholder to run skeleton.

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
    theta_opt = grad_descent(X_train_poly, y_train, lr, gd_iters)

    # Evaluate performance:
    # ... Implement ...
    # ...

    # Plot data:
    plot_data(X_train, class_labels_train)
    plot_data(X_val, class_labels_val)
