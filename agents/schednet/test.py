import random
import numpy as np

def generate_random_numbers(n):
    return np.array([random.random() for _ in range(n)])

# Example usage
n = 10

random_array = generate_random_numbers(n)
print(random_array)