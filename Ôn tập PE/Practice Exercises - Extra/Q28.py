# Question 28: (4 marks)
# Write a program that creates a dictionary using dictionary comprehension with conditions.
# Given a list of numbers, create a dictionary where:
# - Keys are the numbers
# - Values are "even" if the number is even, "odd" if odd
# - Only include numbers divisible by a given divisor
# 
# Input: A list of numbers and a divisor.
# Desired Output: A dictionary with filtered numbers and their parity.
# 
# Example:
# Input: numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], divisor = 2
# Output: {2: 'even', 4: 'even', 6: 'even', 8: 'even', 10: 'even'}
# 
# Input: numbers = [3, 6, 9, 12, 15], divisor = 3
# Output: {3: 'odd', 6: 'even', 9: 'odd', 12: 'even', 15: 'odd'}
# 
# Hints:
# - Use dictionary comprehension
# - Filter by divisibility
# - Check parity with modulo operator

def create_filtered_dict(numbers, divisor):
    """
    TODO: Implement this function
    1. Use dictionary comprehension
    2. Filter numbers divisible by divisor
    3. Assign "even" or "odd" based on parity
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    numbers1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(create_filtered_dict(numbers1, 2))  
    # Expected: {2: 'even', 4: 'even', 6: 'even', 8: 'even', 10: 'even'}
    
    # Test 2
    numbers2 = [3, 6, 9, 12, 15]
    print(create_filtered_dict(numbers2, 3))  
    # Expected: {3: 'odd', 6: 'even', 9: 'odd', 12: 'even', 15: 'odd'}
    
    # Test 3
    numbers3 = [5, 10, 15, 20, 25]
    print(create_filtered_dict(numbers3, 5))  
    # Expected: {5: 'odd', 10: 'even', 15: 'odd', 20: 'even', 25: 'odd'}
