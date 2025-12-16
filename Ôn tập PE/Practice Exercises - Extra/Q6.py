# Question 6: (2 marks)
# Write a program that finds all palindrome words in a given text (ignoring case).
# A palindrome reads the same forwards and backwards.
# Input: A string containing text.
# Desired Output: A list of unique palindrome words (lowercase), in order of first appearance.
# 
# Example:
# Input: "madam Anna racecar hello level world Deed"
# Output: ['madam', 'anna', 'racecar', 'level', 'deed']
# 
# Hints:
# - Convert words to lowercase for comparison
# - Check if word == word[::-1]
# - Use a set to track seen palindromes
# - Preserve order of first appearance

def find_palindromes(text: str) -> list:
    """
    TODO: Implement this function
    1. Split text into words and convert to lowercase
    2. Check each word if it's a palindrome
    3. Keep track of unique palindromes in order
    """
    words = text.split(' ')
    result = []
    seen = set()
    for word in words: 
        word_lower = word.lower()
        if word_lower not in seen and word_lower == word_lower[::-1]: 
            seen.add(word_lower)
            result.append(word_lower)
    return result

# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "madam Anna racecar hello level world Deed"
    print(find_palindromes(text1))  # Expected: ['madam', 'anna', 'racecar', 'level', 'deed']
    
    # Test 2
    text2 = "noon is at noon today"
    print(find_palindromes(text2))  # Expected: ['noon']
    
    # Test 3
    text3 = "no palindrome here"
    print(find_palindromes(text3))  # Expected: []
