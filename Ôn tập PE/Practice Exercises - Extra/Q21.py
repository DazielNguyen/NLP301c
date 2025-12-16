# Question 21: (2 marks)
# Write a program that merges multiple dictionaries, handling key conflicts by summing values.
# Input: A list of dictionaries.
# Desired Output: A single dictionary with merged keys and summed values for conflicts.
# 
# Example:
# Input: [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'a': 2, 'd': 5}]
# Output: {'a': 3, 'b': 5, 'c': 4, 'd': 5}
# 
# Hints:
# - Iterate through each dictionary
# - For each key-value pair, add to result
# - If key exists, sum the values

def merge_dictionaries(dicts):
    """
    TODO: Implement this function
    1. Create an empty result dictionary
    2. Iterate through each dictionary in the list
    3. For each key-value, add or sum to result
    4. Return merged dictionary
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dicts1 = [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'a': 2, 'd': 5}]
    print(merge_dictionaries(dicts1))  # Expected: {'a': 3, 'b': 5, 'c': 4, 'd': 5}
    
    # Test 2
    dicts2 = [{'x': 10}, {'y': 20}, {'z': 30}]
    print(merge_dictionaries(dicts2))  # Expected: {'x': 10, 'y': 20, 'z': 30}
    
    # Test 3
    dicts3 = [{'a': 1, 'b': 1}, {'a': 1, 'b': 1}]
    print(merge_dictionaries(dicts3))  # Expected: {'a': 2, 'b': 2}
