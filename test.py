def convert_to_format(input_list):
    output_list = [[[x // 4, x % 4]] for x in input_list]
    return output_list

# Example usage:
original_list = [2, 6, 3, 15]
new_list = convert_to_format(original_list)
print(new_list)