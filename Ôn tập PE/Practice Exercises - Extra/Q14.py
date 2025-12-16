# Question 14: (3 marks)
# Write a program that merges multiple sorted arrays into one sorted array.
# Input: A list of sorted arrays.
# Desired Output: A single sorted array containing all elements.
# 
# Example:
# Input: [[1, 3, 5], [2, 4, 6], [0, 7, 8]]
# Output: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# 
# Hints:
# - You can simply combine all arrays and sort
# - Or use merge algorithm from merge sort
# - Built-in sorted() function is acceptable

def merge_sorted_arrays(arrays):
    """
    TODO: Implement this function
    1. Combine all arrays into one
    2. Sort the combined array
    3. Return the result
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arrays1 = [[1, 3, 5], [2, 4, 6], [0, 7, 8]]
    print(merge_sorted_arrays(arrays1))  # Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # Test 2
    arrays2 = [[1, 2, 3], [4, 5, 6]]
    print(merge_sorted_arrays(arrays2))  # Expected: [1, 2, 3, 4, 5, 6]
    
    # Test 3
    arrays3 = [[5], [1], [3]]
    print(merge_sorted_arrays(arrays3))  # Expected: [1, 3, 5]
    
    # Test 4
    arrays4 = [[], [1, 2], [3]]
    print(merge_sorted_arrays(arrays4))  # Expected: [1, 2, 3]
