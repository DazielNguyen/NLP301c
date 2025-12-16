# Question 12: (2 marks)
# Write a program that rotates an array to the left or right by k positions.
# Input: A list of elements, k (number of positions), direction ('left' or 'right').
# Desired Output: The rotated array.
# 
# Example:
# Input: arr = [1, 2, 3, 4, 5], k = 2, direction = 'left'
# Output: [3, 4, 5, 1, 2]
# 
# Input: arr = [1, 2, 3, 4, 5], k = 2, direction = 'right'
# Output: [4, 5, 1, 2, 3]
# 
# Hints:
# - Use array slicing
# - Handle k > len(arr) by using k % len(arr)
# - Left rotation: arr[k:] + arr[:k]
# - Right rotation: arr[-k:] + arr[:-k]

def rotate_array(arr, k, direction):
    """
    TODO: Implement this function
    1. Handle edge cases (empty array, k = 0)
    2. Normalize k using modulo
    3. Use slicing based on direction
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 3, 4, 5]
    print(rotate_array(arr1, 2, 'left'))  # Expected: [3, 4, 5, 1, 2]
    
    # Test 2
    arr2 = [1, 2, 3, 4, 5]
    print(rotate_array(arr2, 2, 'right'))  # Expected: [4, 5, 1, 2, 3]
    
    # Test 3
    arr3 = [1, 2, 3]
    print(rotate_array(arr3, 5, 'left'))  # Expected: [3, 1, 2] (k=5 is same as k=2)
    
    # Test 4
    arr4 = [1, 2, 3, 4]
    print(rotate_array(arr4, 0, 'left'))  # Expected: [1, 2, 3, 4]
