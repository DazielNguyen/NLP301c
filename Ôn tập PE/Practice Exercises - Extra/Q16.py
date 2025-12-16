# Question 16: (4 marks)
# Write a program that checks if an array can be split into subarrays with equal sum.
# Input: A list of integers.
# Desired Output: True if the array can be split into equal-sum subarrays, False otherwise.
#                 Return the split point index if True, None if False.
# 
# Example:
# Input: [1, 2, 3, 4, 5, 5]
# Output: (True, 3) - Split at index 3: [1,2,3] sum=6, [4,5,5] sum=14 - wrong example
# Better: [1, 2, 3, 3, 2, 1]
# Output: (True, 3) - Split at index 3: [1,2,3] sum=6, [3,2,1] sum=6
# 
# Hints:
# - Calculate total sum first
# - If total sum is odd, can't split equally
# - Use running sum to find split point
# - Left sum should equal total_sum / 2

def can_split_equal_sum(arr):
    """
    TODO: Implement this function
    1. Calculate total sum
    2. Check if total sum is even
    3. Find split point where left sum = total sum / 2
    4. Return (True, index) or (False, None)
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 3, 3, 2, 1]
    print(can_split_equal_sum(arr1))  # Expected: (True, 3)
    
    # Test 2
    arr2 = [1, 2, 3, 4, 5]
    print(can_split_equal_sum(arr2))  # Expected: (False, None)
    
    # Test 3
    arr3 = [5, 5, 5, 5]
    print(can_split_equal_sum(arr3))  # Expected: (True, 2)
    
    # Test 4
    arr4 = [1, 1, 1, 2]
    print(can_split_equal_sum(arr4))  # Expected: (False, None)
