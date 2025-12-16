# Question 24: (3 marks)
# Write a program that groups items by their values in a dictionary.
# Input: A dictionary.
# Desired Output: A dictionary where keys are unique values and values are lists of keys.
# 
# Example:
# Input: {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2}
# Output: {1: ['a', 'c'], 2: ['b', 'e'], 3: ['d']}
# 
# Hints:
# - Create a new dictionary for grouping
# - Iterate through original dictionary
# - Use value as key, append original key to list

def group_by_value(d):
    """
    TODO: Implement this function
    1. Create empty result dictionary
    2. Iterate through key-value pairs
    3. Group keys by their values
    4. Return grouped dictionary
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2}
    result1 = group_by_value(dict1)
    print(result1)  # Expected: {1: ['a', 'c'], 2: ['b', 'e'], 3: ['d']}
    
    # Test 2
    dict2 = {'x': 'A', 'y': 'B', 'z': 'A'}
    print(group_by_value(dict2))  # Expected: {'A': ['x', 'z'], 'B': ['y']}
    
    # Test 3
    dict3 = {'a': 1, 'b': 2, 'c': 3}
    print(group_by_value(dict3))  # Expected: {1: ['a'], 2: ['b'], 3: ['c']}
