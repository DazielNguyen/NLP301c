# Question 18: (4 marks)
# Write a program that removes duplicates from an array while preserving the order of first appearance.
# Input: A list of elements (may contain duplicates).
# Desired Output: A list with duplicates removed, maintaining original order.
# 
# Example:
# Input: [1, 2, 3, 2, 4, 1, 5]
# Output: [1, 2, 3, 4, 5]
# 
# Input: ['a', 'b', 'a', 'c', 'b']
# Output: ['a', 'b', 'c']
# 
# Hints:
# - Use a set to track seen elements
# - Iterate through array in order
# - Add to result only if not seen before

def remove_duplicates(arr):
    """
    TODO: Implement this function
    1. Create a set to track seen elements
    2. Create a list for result
    3. Iterate through array
    4. Add element if not seen
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 3, 2, 4, 1, 5]
    print(remove_duplicates(arr1))  # Expected: [1, 2, 3, 4, 5]
    
    # Test 2
    arr2 = ['a', 'b', 'a', 'c', 'b']
    print(remove_duplicates(arr2))  # Expected: ['a', 'b', 'c']
    
    # Test 3
    arr3 = [1, 1, 1, 1]
    print(remove_duplicates(arr3))  # Expected: [1]
    
    # Test 4
    arr4 = [5, 4, 3, 2, 1]
    print(remove_duplicates(arr4))  # Expected: [5, 4, 3, 2, 1]
