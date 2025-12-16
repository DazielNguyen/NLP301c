# Question 19: (4 marks)
# Write a program that finds the intersection of multiple arrays.
# Input: A list of arrays.
# Desired Output: A list of elements that appear in all arrays (no duplicates).
# 
# Example:
# Input: [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
# Output: [3, 4]
# 
# Input: [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
# Output: []
# 
# Hints:
# - Convert each array to a set
# - Use set intersection operation
# - You can use & operator or intersection() method
# - Convert result back to list

def find_intersection(arrays):
    """
    TODO: Implement this function
    1. Handle edge cases (empty list, single array)
    2. Convert first array to set
    3. Intersect with remaining arrays
    4. Return as sorted list
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arrays1 = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
    print(sorted(find_intersection(arrays1)))  # Expected: [3, 4]
    
    # Test 2
    arrays2 = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    print(find_intersection(arrays2))  # Expected: []
    
    # Test 3
    arrays3 = [[1, 2], [1, 2], [1, 2]]
    print(sorted(find_intersection(arrays3)))  # Expected: [1, 2]
    
    # Test 4
    arrays4 = [[1, 2, 3]]
    print(sorted(find_intersection(arrays4)))  # Expected: [1, 2, 3]
