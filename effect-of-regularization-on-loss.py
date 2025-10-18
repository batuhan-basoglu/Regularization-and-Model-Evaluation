import numpy as np
import matplotlib.pyplot as plt

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
    print("Running Task 4: Effect of L1 and L2 Regularization on Loss Landscape")

    # Generate dataset
    x, y = generate_linear_data(30)

    # Values of lambda to visualize
    lambda_values = [0.01, 0.1, 1.0]

    # Plot for both L1 and L2 regularization
    for reg_type in ['l1', 'l2']:
        for lam in lambda_values:
            plot_contour(x, y, reg_type, lam)