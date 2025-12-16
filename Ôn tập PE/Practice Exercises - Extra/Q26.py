# Question 26: (4 marks)
# Write a program that flattens a nested dictionary into a single-level dictionary.
# Use dot notation for nested keys.
# Input: A nested dictionary.
# Desired Output: A flattened dictionary with dot-separated keys.
# 
# Example:
# Input: {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
# Output: {'a': 1, 'b.c': 2, 'b.d.e': 3}
# 
# Hints:
# - Use recursion to handle nested dictionaries
# - Build keys by concatenating parent and child keys with '.'
# - Base case: if value is not a dict, add to result

def flatten_dictionary(d, parent_key='', sep='.'):
    """
    TODO: Implement this function
    1. Create empty result dictionary
    2. Iterate through key-value pairs
    3. If value is dict, recursively flatten
    4. Otherwise, add to result with concatenated key
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    print(flatten_dictionary(dict1))  # Expected: {'a': 1, 'b.c': 2, 'b.d.e': 3}
    
    # Test 2
    dict2 = {'x': {'y': {'z': 1}}}
    print(flatten_dictionary(dict2))  # Expected: {'x.y.z': 1}
    
    # Test 3
    dict3 = {'a': 1, 'b': 2}
    print(flatten_dictionary(dict3))  # Expected: {'a': 1, 'b': 2}
    
    # Test 4
    dict4 = {'user': {'name': 'John', 'address': {'city': 'NYC', 'zip': '10001'}}}
    print(flatten_dictionary(dict4))  
    # Expected: {'user.name': 'John', 'user.address.city': 'NYC', 'user.address.zip': '10001'}
