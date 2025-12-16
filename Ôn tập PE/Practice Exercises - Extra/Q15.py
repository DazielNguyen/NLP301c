# Question 15: (3 marks)
# Write a program that finds the missing number in a sequence from 1 to n.
# Input: A list containing n-1 numbers from the range [1, n].
# Desired Output: The missing number.
# 
# Example:
# Input: [1, 2, 4, 5, 6]
# Output: 3
# 
# Input: [2, 3, 4, 5]
# Output: 1
# 
# Hints:
# - Calculate expected sum: n * (n + 1) / 2
# - Calculate actual sum of array
# - Missing number = expected sum - actual sum
# - Or use set difference

def find_missing_number(arr):
    """
    TODO: Implement this function
    1. Determine n (should be len(arr) + 1)
    2. Calculate expected sum
    3. Calculate actual sum
    4. Return the difference
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 4, 5, 6]
    print(find_missing_number(arr1))  # Expected: 3
    
    # Test 2
    arr2 = [2, 3, 4, 5]
    print(find_missing_number(arr2))  # Expected: 1
    
    # Test 3
    arr3 = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    print(find_missing_number(arr3))  # Expected: 8
    
    # Test 4
    arr4 = [1]
    print(find_missing_number(arr4))  # Expected: 2
