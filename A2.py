import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#reproducibility
np.random.seed(1)

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
