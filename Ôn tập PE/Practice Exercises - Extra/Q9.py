# Question 9: (4 marks)
# Write a program that finds the most frequently occurring substring of length k in a given string.
# Input: A string and an integer k (substring length).
# Desired Output: The substring of length k that appears most frequently.
#                 If there's a tie, return the first one encountered.
# 
# Example:
# Input: text = "abcabcabc", k = 3
# Output: "abc"
# 
# Input: text = "aaabbbccc", k = 2
# Output: "aa" (appears first among tied substrings)
# 
# Hints:
# - Generate all substrings of length k
# - Use a dictionary to count frequencies
# - Track the maximum frequency and corresponding substring

def most_frequent_substring(text, k):
    """
    TODO: Implement this function
    1. Create a dictionary to store substring frequencies
    2. Iterate through the string and extract substrings of length k
    3. Count occurrences of each substring
    4. Find and return the most frequent one
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "abcabcabc"
    print(most_frequent_substring(text1, 3))  # Expected: "abc"
    
    # Test 2
    text2 = "aaabbbccc"
    print(most_frequent_substring(text2, 2))  # Expected: "aa" or "bb" or "cc" (first encountered)
    
    # Test 3
    text3 = "hellohello"
    print(most_frequent_substring(text3, 5))  # Expected: "hello"
    
    # Test 4
    text4 = "programming"
    print(most_frequent_substring(text4, 2))  # Expected: "mm"
