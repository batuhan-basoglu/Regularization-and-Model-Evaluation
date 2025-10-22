import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

#reproducibility
np.random.seed(2)

#__________________________________________________________________________________
def true_function(x):
    #return (np.sin(x) + 0.5 * np.cos(3*x) +  0.25 * np.sin(7*x))
    #return np.exp(-0.1 * (x - 5)**2) * np.sin(4 * x)
    #return np.piecewise(x,[x < 4, x >= 4], [lambda x: np.sin(2*x), lambda x: 0.5 * x-3]) 
    return np.log(x + 1) * np.cos(x) + np.sin(2*x)
    
    
    
#Task 1
#1.1
def generate_data(n_samples=100, noise_std=1.0):
    """Generates synthetic data with noise"""
    # generate x values uniformly in [0, 10]
    x = np.linspace(0, 10, n_samples)
    
    #y values without noise
    #y_clean = np.log(x + 1) * np.cos(x) + np.sin(2*x)
    y_clean = true_function(x)
    
    #noise
    noise = np.random.normal(0, noise_std, n_samples)
    y_noisy = y_clean + noise
    
    return x, y_clean, y_noisy

# generate data
x, y_clean, y_noisy = generate_data(100)

# Plot clean and noisy data
plt.figure(figsize=(12, 3))
plt.plot(x, y_clean, 'b-', label='clean data', linewidth=2)
plt.plot(x, y_noisy, 'ro', label='noisy data', alpha=0.6, markersize=4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clean vs noisy data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/task1-data-generation.png')


#__________________________________________________________________________________
#1.2
def gaussian_basis(x, mu, sigma=1.0):
    """Gaussian basis function"""
    return np.exp(-(x - mu)**2 / sigma**2)

def gaussian_features(x, D, sigma=1.0):
    """Create Gaussian basis features"""
    if D == 0:
        return np.ones((len(x), 1))  
    
    x_min, x_max = np.min(x), np.max(x)

    if D == 1:
        mu_i = np.array([(x_min + x_max) / 2])  
    else:
        mu_i = x_min + (x_max - x_min) / (D - 1) * np.arange(D)
    
    features = np.ones((len(x), D + 1))  # with bias term
    
    for i, mu in enumerate(mu_i):
        features[:, i+1] = gaussian_basis(x, mu, sigma).flatten() 
    
    return features


# Plot Gaussian basis functions for different D values
D_values_to_plot = [5, 15, 30,45]
x_plot = np.linspace(0, 10, 200)

plt.figure(figsize=(15, 4))

for i, D in enumerate(D_values_to_plot, 1):
    plt.subplot(1, 4, i)
    
    # Calculate means
    x_min, x_max = np.min(x_plot), np.max(x_plot)
    mu_i = x_min + (x_max - x_min) / (D - 1) * np.arange(D)
    
    # Plot each Gaussian basis
    for mu in mu_i:
        phi = gaussian_basis(x_plot, mu)
        plt.plot(x_plot, phi, alpha=0.7)
    
    plt.title(f'Gaussian basis functions (D={D})')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/task1-non-linear-basis-functions.png')

#__________________________________________________________________________________
#1.3 Model fitting
#for now I used the whole data but idk we that's what they asked for that part
class GaussianRegression:
    """Linear Regression with Gaussian Basis Functions"""
    
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.w = None
        self.D = None
    
    def fit(self, x, y, D):
        # Store D for later use in predict
        self.D = D
        # create features for training and fit using least squares
        X = gaussian_features(x, D, self.sigma)
        #self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        
        return self
    

    def predict(self, x):
        # create features for prediction and predict
        X = gaussian_features(x, self.D, self.sigma)
        yh = X @ self.w
        
        return yh
    



# fit models with different numbers of basis functions and plot
D_values = [0, 5, 10, 13, 15, 17, 20, 25, 30, 45]
x_plot = np.linspace(0, 10, 300) 

plt.figure(figsize=(20, 8))

for i, D in enumerate(D_values):
    plt.subplot(2, 5, i+1)

    # Create new model for each D value, fit and get predictions 
    model = GaussianRegression(sigma=1.0)
    model.fit(x, y_noisy, D)
    y_hat = model.predict(x_plot)
    
    # Ensure y_hat is 1D and has same length as x_plot
    y_hat = y_hat.flatten() if y_hat.ndim > 1 else y_hat
    
    # Plot
    plt.plot(x_plot, true_function(x_plot), 'b-', label='True function', linewidth=2, alpha=0.7)
    plt.plot(x, y_noisy, 'ro', label='Noisy data', alpha=0.4, markersize=3)
    plt.plot(x_plot, y_hat, 'g-', label=f'Fitted (D={D})', linewidth=2)
    
    plt.ylim(-4.2, 4.2)
    plt.title(f'D = {D}')
    plt.grid(True, alpha=0.3)
    if D == 0:
        plt.legend(fontsize=8)

    # x and y labels
    if i % 5 == 0:  
        plt.ylabel('y')
    if i >= 5:  
        plt.xlabel('x')

plt.tight_layout()
plt.savefig('results/task1-model-fitting.png')

#__________________________________________________________________________________
#1.4 Model Selection

# Split the data into training and validation sets 
x_train, x_val, y_train, y_val = train_test_split(x, y_noisy, test_size=0.3, random_state=100)

# range of basis functions to test
D_values = list(range(0, 46))  # 0 to 45

# Initialize arrays to store errors
train_sse = []
val_sse = []


# For each number of basis functions
for D in D_values:
    # Create and fit the model
    model = GaussianRegression(sigma=1.0)
    model.fit(x_train, y_train, D)
    
    # predict on training and validation set
    yh_train = model.predict(x_train)
    yh_val = model.predict(x_val)
    
    # compute SSE
    sse_train = np.sum((y_train - yh_train)**2)
    sse_val = np.sum((y_val - yh_val)**2)
    
    train_sse.append(sse_train)
    val_sse.append(sse_val)
    
    print(f"D={D}: Train SSE={sse_train:.0f}, Val SSE={sse_val:.0f}")


optimal_D = D_values[int(np.argmin(val_sse))]
print(f"Optimal D on single split = {optimal_D}")
#optimal_sse = np.min(val_sse)
#MAYBE CAN ADD A MANUAL LOWER BOUND 


# Plot training and validation SSE vs D for this single split
plt.figure(figsize=(12, 3))
plt.plot(D_values, train_sse, 'b-', label='Train SSE', linewidth=2, marker='o', markersize=4)
plt.plot(D_values, val_sse, 'r-', label='Validation SSE', linewidth=2, marker='s', markersize=4)
plt.axvline(x=optimal_D, color='g', linestyle='--', label=f'Optimal D = {optimal_D}')
#plt.scatter([optimal_D], [val_sse[optimal_D]], label=f"Opt D = {optimal_D}", zorder=5)
plt.xlabel('Number of gaussian bases (D)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Train and validation SSE vs D (single split)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log') 
plt.savefig('results/task1-model-selection-single-split.png')


# plot optimal model fit
plt.figure(figsize=(10, 3))
optimal_model = GaussianRegression(sigma=1.0)
yh_opt = optimal_model.fit(x_train, y_train, optimal_D)
yh_opt = optimal_model.predict(x_plot)

plt.plot(x_plot, true_function(x_plot), 'b-', label='true function', linewidth=2)
plt.plot(x_train, y_train, 'bo', label='Training data', alpha=0.6, markersize=4)
plt.plot(x_val, y_val, 'ro', label='validation data', alpha=0.6, markersize=4)
plt.plot(x_plot, yh_opt, 'g-', label=f'Optimal Model (D={optimal_D})', linewidth=2)
plt.ylim(-5, 5)
plt.title(f'Optimal model with {optimal_D} gaussian basis functions')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/task1-model-selection-optimal-model.png')

#__________________________________________________________________________________
#2. Bias-Variance Tradeoff with Multiple Fits
#sigma = (x.max() - x.min()) / D

n_repetitions = 10
D_values = [0, 5, 7, 10, 12, 15, 20, 25, 30, 45]
x = np.linspace(0, 10, 300)

# Initialize arrays to store results
train_errs = np.zeros((n_repetitions, len(D_values)))
test_errs = np.zeros((n_repetitions, len(D_values)))
predictions = np.zeros((n_repetitions, len(D_values), len(x)))

for rep in range(n_repetitions):
    #create new dataset
    x_data, y_true, y_data = generate_data(100, noise_std=1.0)
    
    # split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=rep)

    for D_i, D in enumerate(D_values):
        # fit model
        '''
        if D > 10:
            sigma = (x.max() - x.min()) / D
        else: 
            sigma = 1
        '''
        sigma = 1
        model = GaussianRegression(sigma)
        #model = GaussianRegression(sigma=1.0)
        model.fit(x_train, y_train, D) 

        # predict on both sets 
        yh_train = model.predict(x_train)  
        yh_test = model.predict(x_test)    
        
        # compute and store errors (MSE)
        train_err = np.mean((y_train - yh_train)**2)
        test_err = np.mean((y_test - yh_test)**2)
        
        train_errs[rep, D_i] = train_err
        test_errs[rep, D_i] = test_err

        # predict for visualization
        yh_cont = model.predict(x)
        predictions[rep, D_i, :] = yh_cont


# Plot 1: fitted models on the same plot, bias-variance tradeoff visualization
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for D_i, D in enumerate(D_values):
    ax = axes[D_i]
    
    # plot individual fits
    for rep in range(n_repetitions):
        if train_errs[rep, D_i] != np.inf:
            ax.plot(x, predictions[rep, D_i, :], color='green', alpha=0.3, linewidth=1)
    
    # plot true function
    ax.plot(x, true_function(x), 'b-', linewidth=3, label='true function')
    
    #plot average prediction
    valid_predictions = [predictions[rep, D_i, :] 
                        for rep in range(n_repetitions) 
                        if train_errs[rep, D_i] != np.inf]
    
    if valid_predictions:
        avg_prediction = np.mean(valid_predictions, axis=0)
        ax.plot(x, avg_prediction, 'r-', linewidth=2, label='Average prediction')
    
    ax.set_title(f'D = {D}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    
    if D_i == 0:
        ax.legend()

plt.tight_layout()
plt.suptitle('Bias-Variance tradeoff with 10 different fits', fontsize=16, y=1.02)
plt.savefig('results/task2-plotting-multiple-fits.png')

# Plot 2: average training and test errors
plt.figure(figsize=(12, 4))

# Compute mean and std
avg_train_errors = np.mean(train_errs, axis=0)
avg_test_errors = np.mean(test_errs, axis=0)
std_train_errors = np.std(train_errs, axis=0)
std_test_errors = np.std(test_errs, axis=0)


# Plot with error bars
plt.errorbar(D_values, avg_train_errors, yerr=std_train_errors, label='Average Training Error', marker='o', capsize=5, linewidth=2)
plt.errorbar(D_values, avg_test_errors, yerr=std_test_errors, label='Average Test Error', marker='s', capsize=5, linewidth=2)

plt.xlabel('number of gaussian basis functions (D)')
plt.ylabel('Mean Squared Error')
plt.title('Average training and test errors across 10 repetitions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xticks(D_values)
plt.savefig('results/task2-plotting-train-and-test-errors.png')
