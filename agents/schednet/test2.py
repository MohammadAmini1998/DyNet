import itertools

def generate_action_space():
        values = [0, 1, 2, 3,4,5]
        action_space = []

        # Generate all possible combinations of values
        combinations = list(itertools.product(values, repeat=4))

        # Filter combinations where the sum of each element is not above 3
        valid_combinations = [combo for combo in combinations if sum(combo) <=5 and sum(combo)!=0]

        # Convert combinations to action vectors
        for combo in valid_combinations:
            action_space.append(list(combo))

        return action_space
print((generate_action_space()))