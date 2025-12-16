# Question 11: (2 marks)
# Write a program that finds all pairs of numbers in an array that sum to a target value.
# Input: A list of integers and a target sum.
# Desired Output: A list of tuples, each containing a pair of numbers that sum to target.
#                 Return unique pairs (order doesn't matter).
# 
# Example:
# Input: arr = [1, 2, 3, 4, 5], target = 5
# Output: [(1, 4), (2, 3)]
# 
# Input: arr = [2, 7, 11, 15], target = 9
# Output: [(2, 7)]
# 
# Hints:
# - Use a set to track seen numbers
# - For each number, check if (target - number) exists in the set
# - Avoid duplicate pairs

def find_pairs_with_sum(arr, target):
    """
    TODO: Implement this function
    1. Create a set to track seen numbers
    2. Create a set to track found pairs (to avoid duplicates)
    3. For each number, check if complement exists
    4. Add valid pairs to result
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 3, 4, 5]
    print(find_pairs_with_sum(arr1, 5))  # Expected: [(1, 4), (2, 3)]
    
    # Test 2
    arr2 = [2, 7, 11, 15]
    print(find_pairs_with_sum(arr2, 9))  # Expected: [(2, 7)]
    
    # Test 3
    arr3 = [1, 1, 1, 1]
    print(find_pairs_with_sum(arr3, 2))  # Expected: [(1, 1)]
    
    # Test 4
    arr4 = [1, 2, 3]
    print(find_pairs_with_sum(arr4, 10))  # Expected: []
