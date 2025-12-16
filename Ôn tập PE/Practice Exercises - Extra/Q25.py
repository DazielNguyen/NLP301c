# Question 25: (3 marks)
# Write a program that counts the frequency of values in a dictionary.
# Input: A dictionary.
# Desired Output: A dictionary showing how many times each value appears.
# 
# Example:
# Input: {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2, 'f': 1}
# Output: {1: 3, 2: 2, 3: 1}
# 
# Hints:
# - Create a frequency dictionary
# - Iterate through values
# - Count occurrences of each value

def value_frequency(d):
    """
    TODO: Implement this function
    1. Create empty frequency dictionary
    2. Iterate through values
    3. Count each value occurrence
    4. Return frequency dictionary
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2, 'f': 1}
    print(value_frequency(dict1))  # Expected: {1: 3, 2: 2, 3: 1}
    
    # Test 2
    dict2 = {'x': 'A', 'y': 'B', 'z': 'A', 'w': 'A'}
    print(value_frequency(dict2))  # Expected: {'A': 3, 'B': 1}
    
    # Test 3
    dict3 = {'a': 5, 'b': 5, 'c': 5}
    print(value_frequency(dict3))  # Expected: {5: 3}
