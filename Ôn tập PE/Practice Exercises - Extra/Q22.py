# Question 22: (2 marks)
# Write a program that inverts a dictionary (swap keys and values).
# If multiple keys have the same value, keep the last one encountered.
# Input: A dictionary.
# Desired Output: A dictionary with keys and values swapped.
# 
# Example:
# Input: {'a': 1, 'b': 2, 'c': 3}
# Output: {1: 'a', 2: 'b', 3: 'c'}
# 
# Input: {'a': 1, 'b': 1, 'c': 2}
# Output: {1: 'b', 2: 'c'} (last key 'b' with value 1 is kept)
# 
# Hints:
# - Iterate through items
# - Create new dict with value as key, key as value
# - Later entries will overwrite earlier ones with same value

def invert_dictionary(d):
    """
    TODO: Implement this function
    1. Create empty result dictionary
    2. Iterate through key-value pairs
    3. Add value as key and key as value
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    print(invert_dictionary(dict1))  # Expected: {1: 'a', 2: 'b', 3: 'c'}
    
    # Test 2
    dict2 = {'a': 1, 'b': 1, 'c': 2}
    print(invert_dictionary(dict2))  # Expected: {1: 'b', 2: 'c'} or {1: 'a', 2: 'c'}
    
    # Test 3
    dict3 = {'apple': 'fruit', 'carrot': 'vegetable'}
    print(invert_dictionary(dict3))  # Expected: {'fruit': 'apple', 'vegetable': 'carrot'}
