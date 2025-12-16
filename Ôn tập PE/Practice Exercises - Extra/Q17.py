# Question 17: (4 marks)
# Write a program that finds the length of the longest consecutive sequence in an unsorted array.
# Input: A list of integers (may contain duplicates).
# Desired Output: The length of the longest consecutive sequence.
# 
# Example:
# Input: [100, 4, 200, 1, 3, 2]
# Output: 4 (sequence: 1, 2, 3, 4)
# 
# Input: [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
# Output: 9 (sequence: 0, 1, 2, 3, 4, 5, 6, 7, 8)
# 
# Hints:
# - Convert array to set for O(1) lookup
# - For each number, check if it's the start of a sequence
# - A number is start if (number - 1) not in set
# - Count consecutive numbers from each start

def longest_consecutive_sequence(arr):
    """
    TODO: Implement this function
    1. Convert array to set to remove duplicates
    2. For each number, check if it starts a sequence
    3. Count consecutive numbers
    4. Track maximum length
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [100, 4, 200, 1, 3, 2]
    print(longest_consecutive_sequence(arr1))  # Expected: 4
    
    # Test 2
    arr2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    print(longest_consecutive_sequence(arr2))  # Expected: 9
    
    # Test 3
    arr3 = [1, 2, 0, 1]
    print(longest_consecutive_sequence(arr3))  # Expected: 3
    
    # Test 4
    arr4 = [9, 1, 4, 7, 3]
    print(longest_consecutive_sequence(arr4))  # Expected: 2 (1,2,3 or 3,4)
