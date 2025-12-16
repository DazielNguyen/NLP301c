# Question 13: (3 marks)
# Write a program that finds the element that appears most frequently in an array.
# If there's a tie, return the element that appears first.
# Input: A list of elements.
# Desired Output: The most frequent element.
# 
# Example:
# Input: [1, 2, 2, 3, 3, 3, 4]
# Output: 3
# 
# Input: [1, 1, 2, 2, 3]
# Output: 1 (appears first among tied elements)
# 
# Hints:
# - Use a dictionary to count frequencies
# - Track the maximum frequency
# - For ties, keep the first encountered element

def most_frequent_element(arr):
    """
    TODO: Implement this function
    1. Create a dictionary to count frequencies
    2. Track max frequency and corresponding element
    3. Iterate through array to maintain order for ties
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, 2, 2, 3, 3, 3, 4]
    print(most_frequent_element(arr1))  # Expected: 3
    
    # Test 2
    arr2 = [1, 1, 2, 2, 3]
    print(most_frequent_element(arr2))  # Expected: 1
    
    # Test 3
    arr3 = ['a', 'b', 'c', 'a', 'b', 'a']
    print(most_frequent_element(arr3))  # Expected: 'a'
    
    # Test 4
    arr4 = [5]
    print(most_frequent_element(arr4))  # Expected: 5
