import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

#reproducibility
np.random.seed(2)

#__________________________________________________________________________________
#Task 1
#1.1
def generate_data(n_samples=100, noise_std=1.0):
    """Generates synthetic data with noise"""
    # generate x values uniformly in [0, 10]
    x = np.linspace(0, 10, n_samples)
    
    #y values without noise
    y_clean = (np.log(x + 1e-10) + 1) * np.cos(x) + np.sin(2*x)
    
    #noise
    noise = np.random.normal(0, noise_std, n_samples)
    y_noisy = y_clean + noise
    
    return x, y_clean, y_noisy

# generate data
x, y_clean, y_noisy = generate_data(100)

# Plot clean and noisy data
plt.plot(x, y_clean, 'b-', label='Clean Data', linewidth=2)
plt.plot(x, y_noisy, 'ro', label='Noisy Data', alpha=0.6, markersize=4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clean vs Noisy Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


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
    mu_i = x_min + (x_max - x_min) / (D - 1) * np.arange(D)
    
    features = np.ones((len(x), D + 1))  # with bias term
    
    for i, mu in enumerate(mu_i):
        features[:, i+1] = gaussian_basis(x, mu, sigma)
    
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
    
    plt.title(f'Gaussian Basis Functions (D={D})')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        
        return self
    

    def predict(self, x):
        # create features for prediction and predict
        X = gaussian_features(x, self.D, self.sigma)
        yh = X @ self.w
        
        return yh
    
    

def true_function(x):
    return (np.log(x + 1e-10) + 1) * np.cos(x) + np.sin(2*x)

# fit models with different numbers of basis functions and plot
D_i = [0, 2, 5, 10, 13, 15, 17, 20, 25, 30, 35, 45]
x_plot = np.linspace(0, 10, 300) 

plt.figure(figsize=(18, 12))

for i, D in enumerate(D_i):
    plt.subplot(4, 3, i+1)
    
    # Create new model for each D value, fit and get predictions 
    model = GaussianRegression(sigma=1.0)
    model.fit(x, y_noisy, D)
    y_hat = model.predict(x_plot)
    
    # Ensure y_hat is 1D and has same length as x_plot
    y_hat = y_hat.flatten() if y_hat.ndim > 1 else y_hat
    
    # Plot
    plt.plot(x_plot, true_function(x_plot), 'b-', label='True Function', linewidth=2, alpha=0.7)
    plt.plot(x, y_noisy, 'ro', label='Noisy Data', alpha=0.4, markersize=3)
    plt.plot(x_plot, y_hat, 'g-', label=f'Fitted (D={D})', linewidth=2)
    
    plt.ylim(-6, 6)
    plt.title(f'D = {D}')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    # x and y labels
    if i % 3 == 0:  
        plt.ylabel('y')
    if i >= 9:  
        plt.xlabel('x')

plt.tight_layout()
plt.show()

#__________________________________________________________________________________
#1.4 Model Selection

# Split the data into training and validation sets 
x_train, x_val, y_train, y_val = train_test_split(x, y_noisy, test_size=0.2, random_state=42)

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
    
    # predict on training then validation 
    yh_train = model.predict(x_train)
    yh_train = yh_train.flatten() if yh_train.ndim > 1 else yh_train
    
    yh_val = model.predict(x_val)
    yh_val = yh_val.flatten() if yh_val.ndim > 1 else yh_val
    
    # compute SSE
    sse_train = np.sum((y_train - yh_train)**2)
    sse_val = np.sum((y_val - yh_val)**2)
    
    train_sse.append(sse_train)
    val_sse.append(sse_val)
    
    print(f"D={D:2d}: Train SSE = {sse_train:8.2f}, Val SSE = {sse_val:8.2f}")




