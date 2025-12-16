# Question 20: (4 marks)
# Write a program that finds the maximum value in each sliding window of size k.
# Input: An array of numbers and window size k.
# Desired Output: A list of maximum values for each window position.
# 
# Example:
# Input: arr = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
# Output: [3, 3, 5, 5, 6, 7]
# Explanation:
# Window [1, 3, -1] -> max = 3
# Window [3, -1, -3] -> max = 3
# Window [-1, -3, 5] -> max = 5
# Window [-3, 5, 3] -> max = 5
# Window [5, 3, 6] -> max = 6
# Window [3, 6, 7] -> max = 7
# 
# Hints:
# - Iterate through array with window of size k
# - For each window, find maximum
# - Use max() function on slices

def sliding_window_maximum(arr, k):
    """
    TODO: Implement this function
    1. Handle edge cases (k > len(arr))
    2. Iterate from 0 to len(arr) - k + 1
    3. For each position, get max of window
    4. Append to result
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 3, -1, -3, 5, 3, 6, 7]
    print(sliding_window_maximum(arr1, 3))  # Expected: [3, 3, 5, 5, 6, 7]
    
    # Test 2
    arr2 = [1, 2, 3, 4, 5]
    print(sliding_window_maximum(arr2, 2))  # Expected: [2, 3, 4, 5]
    
    # Test 3
    arr3 = [5, 4, 3, 2, 1]
    print(sliding_window_maximum(arr3, 3))  # Expected: [5, 4, 3]
    
    # Test 4
    arr4 = [1, 1, 1, 1, 1]
    print(sliding_window_maximum(arr4, 2))  # Expected: [1, 1, 1, 1]
