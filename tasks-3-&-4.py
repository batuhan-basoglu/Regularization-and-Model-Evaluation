import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Linear Regression Data Generation
def generate_data(N, noise_var):
    x = np.random.uniform(0, 1, N)
    noise = np.random.normal(0, np.sqrt(noise_var), N)
    y = np.sin(2 * np.pi * x) + noise
    return x.reshape(-1, 1), y

# Gaussian Basis is used for the linear regression
def gaussian_basis(x, D):
    mus = np.linspace(0, 1, D) # means initialized.
    s = 0.1 # standard deviation
    basis = np.exp(-(x - mus)**2 / (2 * s**2)) # Gaussian Formula
    return basis

# Ridge Regression Formula
def l2_ridge_regression(x, y, lam):
    I = np.eye(x.shape[1]) # regularization term
    return inv(x.T @ x + lam * I) @ x.T @ y # (X^T*X+λ*I)^−1*X^T*y

# Lasso Regression Formula
def l1_lasso_regression(x, y, lam, lr=0.01, iterations=1000):
    w = np.zeros(x.shape[1]) # w initialized the weight
    for i in range(iterations): # iterations
        grad = -x.T @ (y - x @ w) # gradient formula
        w -= lr * grad # gradient descent
        w = np.sign(w) * np.maximum(0, np.abs(w) - lr * lam) # soft thresholding
    return w

# Cross Validation
def cross_validate(x, y, lam_values, reg_type, k):
    fold_size = len(x) // k # fold size is defined as 10
    train_errors, val_errors = [], []

    for lam in lam_values:
        train_mse, val_mse = [], []
        for i in range(k): # trained for 9 folds and validated in the remaining with each λ.
            val_idx = np.arange(i * fold_size, (i + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(len(x)), val_idx)
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # apply the regression
            if reg_type == 'l2':
                w = l2_ridge_regression(x_train, y_train, lam)
            else:
                w = l1_lasso_regression(x_train, y_train, lam)

            train_pred = x_train @ w
            val_pred = x_val @ w
            train_mse.append(np.mean((y_train - train_pred)**2)) # train mse added for each fold
            val_mse.append(np.mean((y_val - val_pred)**2)) # validation mse added for each fold

        train_errors.append(np.mean(train_mse)) # train error added for each λ
        val_errors.append(np.mean(val_mse)) # validation error added for each λ
    return train_errors, val_errors

# Train and Validation Errors
def train_validation_err(reg_type, lam_values, num_datasets, N, D):
    if lam_values is None:
        lam_values = np.logspace(-3, 1, 10) # λ is defined

    train_all, val_all = [], []

    for i in range(num_datasets): # inside given 50 datasets
        x, y = generate_data(N, 1.0) # data generated
        Phi = gaussian_basis(x, D) # linear regression with gaussian basis
        train_err, val_err = cross_validate(Phi, y, lam_values, reg_type, 10) # cross-validation training
        train_all.append(train_err)
        val_all.append(val_err)

    mean_train = np.mean(train_all, 0) # mean training
    mean_val = np.mean(val_all, 0) # mean validation

    # plotting the figure
    plt.figure()
    plt.plot(lam_values, mean_train, label='Train Error')
    plt.plot(lam_values, mean_val, label='Validation Error')
    plt.xscale('log')
    plt.xlabel("λ")
    plt.ylabel("MSE")
    plt.title(f"{reg_type.upper()} Regularization")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task3-train-validation-errors-' + reg_type + '.png')

    return mean_train, mean_val, lam_values

# Bias-Variance Decomposition
def bias_variance_decomp(reg_type, lam_values, num_datasets, N, D):
    if lam_values is None:
        lam_values = np.logspace(-3, 1, 10) # λ is defined

    x_test = np.linspace(0, 1, 100).reshape(-1, 1) # x test value defined
    Phi_test = gaussian_basis(x_test, D) # linear regression with gaussian basis
    y_true = np.sin(2 * np.pi * x_test).ravel() # sin(2*π*x)

    biases, variances, total_mse = [], [], []

    for lam in lam_values:
        preds = []
        for i in range(num_datasets):
            x_train, y_train = generate_data(N, 1.0) # data generated
            Phi_train = gaussian_basis(x_train, D) # linear regression with gaussian basis
            # apply the regression
            if reg_type == 'l2':
                w = l2_ridge_regression(Phi_train, y_train, lam)
            else:
                w = l1_lasso_regression(Phi_train, y_train, lam)
            preds.append(Phi_test @ w)

        preds = np.array(preds)
        mean_pred = np.mean(preds, 0) # mean of predictions
        bias2 = np.mean((mean_pred - y_true) ** 2) # squared bias formula
        # bias is defined as mean of difference between predicted mean and true y value.
        var = np.mean(np.var(preds, 0)) # mean variance of predictions
        mse = bias2 + var + 1  # noise variance

        biases.append(bias2) # add bias for each λ
        variances.append(var) # add variance for each λ
        total_mse.append(mse) # add total mse for each λ

    # plotting the figure
    plt.figure()
    plt.plot(lam_values, biases, label='Bias^2')
    plt.plot(lam_values, variances, label='Variance')
    plt.plot(lam_values, np.array(biases) + np.array(variances), label='Bias^2 + Var')
    plt.plot(lam_values, total_mse, label='Bias^2 + Var + Noise')
    plt.xscale('log')
    plt.xlabel("λ")
    plt.ylabel("Error")
    plt.title(f"{reg_type.upper()} Bias-Variance Decomposition")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task3-bias-decomposition-' + reg_type + '.png')

# Generating Synthetic Data
def generate_linear_data(n):
    x = np.random.uniform(0, 10, n) # initialize x
    eps = np.random.normal(0, 1, n) # initialize epsilon
    y = -3 * x + 8 + 2 * eps # y = −3x + 8 + 2ϵ
    return x.reshape(-1, 1), y

# Gradient Descent with L1/L2
def gradient_descent(x, y, lam, reg_type, lr, iters):
    x_b = np.hstack([np.ones_like(x), x]) # initialize x
    w = np.zeros(x_b.shape[1]) # initialize weight
    path = [w.copy()]

    for i in range(iters):
        pred = x_b @ w # linear regression prediction
        error = pred - y # error
        grad = x_b.T @ error / len(y) # gradient formula

        if reg_type == 'l2':
            grad += lam * w # L2 formula
        elif reg_type == 'l1':
            grad += lam * np.sign(w) # L1 formula

        w -= lr * grad # loss calculation
        path.append(w.copy())

    return w, np.array(path)

# Plotting the loss
def plot_contour(x, y, reg_type, lam):
    x_b = np.hstack([np.ones_like(x), x]) # initialize x
    w0, w1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)) # initialize intercept and slope
    loss = np.zeros_like(w0) # initialize loss

    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            w = np.array([w0[i, j], w1[i, j]])
            error = y - x_b @ w # error
            mse = np.mean(error ** 2) # mean square error
            reg = lam * (np.sum(w ** 2) if reg_type == 'l2' else np.sum(np.abs(w))) # regularization
            loss[i, j] = mse + reg # regularization and mse for the loss

    _, path = gradient_descent(x, y, lam, reg_type, 0.01, 500)

    # plotting the figure
    plt.figure(figsize=(6, 5))
    plt.contour(w0, w1, loss, levels=50, cmap='viridis')
    plt.plot(path[:, 0], path[:, 1], 'ro-', markersize=2, label='Gradient Descent Path')
    plt.title(f"{reg_type.upper()} Regularization (λ={lam})")
    plt.xlabel("w0 (intercept)")
    plt.ylabel("w1 (slope)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/task4-effect-of-regularization-on-loss-' + reg_type + '-'  + str(lam) + '.png')


if __name__ == "__main__":
    print("Running Task 3: Regularization with Cross-Validation")

    lam_values = np.logspace(-3, 1, 10) # initial λ values -3, 1, 10

    train_validation_err('l2', lam_values, 50, 20, 45)
    train_validation_err('l1', lam_values, 50, 20, 45)

    bias_variance_decomp('l2', lam_values, 50, 20, 45)
    bias_variance_decomp('l1', lam_values, 50, 20, 45)

    print("Running Task 4: Effect of L1 and L2 Regularization on Loss Landscape")

    # Generate dataset
    x, y = generate_linear_data(30)

    # Values of lambda to visualize
    lambda_values = [0.01, 0.1, 1.0]

    # Plot for both L1 and L2 regularization
    for reg_type in ['l1', 'l2']:
        for lam in lambda_values:
            plot_contour(x, y, reg_type, lam)