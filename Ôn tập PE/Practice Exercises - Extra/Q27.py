# Question 27: (4 marks)
# Write a program that finds the top K keys with the largest values in a dictionary.
# Input: A dictionary and an integer K.
# Desired Output: A list of tuples (key, value) for top K items, sorted by value descending.
# 
# Example:
# Input: {'a': 5, 'b': 2, 'c': 8, 'd': 1, 'e': 7}, K = 3
# Output: [('c', 8), ('e', 7), ('a', 5)]
# 
# Hints:
# - Convert dictionary to list of tuples
# - Sort by value in descending order
# - Return first K items

def top_k_items(d, k):
    """
    TODO: Implement this function
    1. Convert dictionary items to list of tuples
    2. Sort by value in descending order
    3. Return first k items
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    dict1 = {'a': 5, 'b': 2, 'c': 8, 'd': 1, 'e': 7}
    print(top_k_items(dict1, 3))  # Expected: [('c', 8), ('e', 7), ('a', 5)]
    
    # Test 2
    dict2 = {'x': 10, 'y': 20, 'z': 15}
    print(top_k_items(dict2, 2))  # Expected: [('y', 20), ('z', 15)]
    
    # Test 3
    dict3 = {'a': 1, 'b': 2, 'c': 3}
    print(top_k_items(dict3, 5))  # Expected: [('c', 3), ('b', 2), ('a', 1)]
    
    # Test 4
    dict4 = {'python': 100, 'java': 80, 'javascript': 90, 'c++': 70}
    print(top_k_items(dict4, 2))  # Expected: [('python', 100), ('javascript', 90)]
