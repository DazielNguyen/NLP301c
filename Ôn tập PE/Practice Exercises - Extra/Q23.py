# Question 23: (3 marks)
# Write a program that filters a dictionary based on a condition.
# Input: A dictionary and a filter type ('even_values', 'odd_values', 'positive_values', 'negative_values').
# Desired Output: A dictionary containing only items that meet the condition.
# 
# Example:
# Input: {'a': 2, 'b': 3, 'c': 4, 'd': 5}, filter_type = 'even_values'
# Output: {'a': 2, 'c': 4}
# 
# Input: {'x': -1, 'y': 2, 'z': -3}, filter_type = 'positive_values'
# Output: {'y': 2}
# 
# Hints:
# - Use dictionary comprehension
# - Check value based on filter_type
# - Return filtered dictionary

def filter_dictionary(d, filter_type):
    """
    TODO: Implement this function
    1. Check filter_type
    2. Use appropriate condition for filtering
    3. Return dictionary with matching items
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
    print(filter_dictionary(dict1, 'even_values'))  # Expected: {'a': 2, 'c': 4}
    
    # Test 2
    dict2 = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
    print(filter_dictionary(dict2, 'odd_values'))  # Expected: {'b': 3, 'd': 5}
    
    # Test 3
    dict3 = {'x': -1, 'y': 2, 'z': -3}
    print(filter_dictionary(dict3, 'positive_values'))  # Expected: {'y': 2}
    
    # Test 4
    dict4 = {'x': -1, 'y': 2, 'z': -3}
    print(filter_dictionary(dict4, 'negative_values'))  # Expected: {'x': -1, 'z': -3}
