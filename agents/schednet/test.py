
import numpy as np

# Function to calculate entropy
def calculate_entropy(data):
    # Discretize the data into bins
    num_bins = 1000
    counts, bins = np.histogram(data, bins=num_bins)
    
    # Calculate probabilities
    probabilities = counts / len(data)
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add a small value to avoid log(0)
    
    return entropy

# Sample continuous temperature data
temperature_data = np.random.uniform(low=0, high=100, size=1000)

print(temperature_data)
# Calculate entropy
entropy = calculate_entropy(temperature_data)

print("Entropy of temperature data:", entropy)