import itertools
import numpy as np

def generate_action_space():
    values = [0, 1, 2, 3]
    action_space = []
    action_sums = []

    combinations = list(itertools.product(values, repeat=4))
    valid_combinations = [combo for combo in combinations if sum(combo) <= 3]

    for combo in valid_combinations:
        if sum(combo) == 3:
            
            action_space.append(list(combo))
            action_sums.append(sum(combo))

    return action_space, action_sums

a, num = generate_action_space()
print(a[0])