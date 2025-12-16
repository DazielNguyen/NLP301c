# Question 10: (4 marks)
# Write a program that removes consecutive duplicate characters from a string.
# Input: A string.
# Desired Output: A string with consecutive duplicates removed.
# 
# Example:
# Input: "aabbccddee"
# Output: "abcde"
# 
# Input: "hellooo world!!!"
# Output: "helo world!"
# 
# Hints:
# - Iterate through the string
# - Compare each character with the previous one
# - Only add if it's different from the previous character

def remove_consecutive_duplicates(text):
    """
    TODO: Implement this function
    1. Handle empty string case
    2. Start with first character
    3. Iterate from second character onwards
    4. Add character only if different from previous
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "aabbccddee"
    print(remove_consecutive_duplicates(text1))  # Expected: "abcde"
    
    # Test 2
    text2 = "hellooo world!!!"
    print(remove_consecutive_duplicates(text2))  # Expected: "helo world!"
    
    # Test 3
    text3 = "aaaa"
    print(remove_consecutive_duplicates(text3))  # Expected: "a"
    
    # Test 4
    text4 = "abcdef"
    print(remove_consecutive_duplicates(text4))  # Expected: "abcdef"
    
    # Test 5
    text5 = ""
    print(remove_consecutive_duplicates(text5))  # Expected: ""
