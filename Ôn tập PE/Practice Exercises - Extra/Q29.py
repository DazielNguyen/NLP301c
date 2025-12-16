# Question 29: (4 marks)
# Write a program that compares two dictionaries and finds the differences.
# Input: Two dictionaries.
# Desired Output: A dictionary with three keys:
#   - 'added': keys in dict2 but not in dict1
#   - 'removed': keys in dict1 but not in dict2
#   - 'changed': keys in both with different values
# 
# Example:
# Input: dict1 = {'a': 1, 'b': 2, 'c': 3}, dict2 = {'b': 2, 'c': 4, 'd': 5}
# Output: {
#     'added': {'d': 5},
#     'removed': {'a': 1},
#     'changed': {'c': (3, 4)}
# }
# 
# Hints:
# - Use set operations on keys
# - Compare values for common keys
# - Build result dictionary with three categories

def compare_dictionaries(dict1, dict2):
    """
    TODO: Implement this function
    1. Find keys only in dict1 (removed)
    2. Find keys only in dict2 (added)
    3. Find common keys with different values (changed)
    4. Return result as dictionary
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    dict2 = {'b': 2, 'c': 4, 'd': 5}
    print(compare_dictionaries(dict1, dict2))
    # Expected: {'added': {'d': 5}, 'removed': {'a': 1}, 'changed': {'c': (3, 4)}}
    
    # Test 2
    dict1 = {'x': 10, 'y': 20}
    dict2 = {'x': 10, 'y': 20}
    print(compare_dictionaries(dict1, dict2))
    # Expected: {'added': {}, 'removed': {}, 'changed': {}}
    
    # Test 3
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    print(compare_dictionaries(dict1, dict2))
    # Expected: {'added': {'b': 2}, 'removed': {'a': 1}, 'changed': {}}
