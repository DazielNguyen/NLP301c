# Question 8: (3 marks)
# Write a program that counts the number of vowels and consonants in a string 
# (ignoring non-alphabetic characters).
# Input: A string.
# Desired Output: A dictionary with keys 'vowels' and 'consonants' and their counts.
# 
# Example:
# Input: "Hello World 123!"
# Output: {'vowels': 3, 'consonants': 7}
# 
# Hints:
# - Define vowels as 'aeiouAEIOU'
# - Filter only alphabetic characters
# - Count vowels and consonants separately

def count_vowels_consonants(text):
    """
    TODO: Implement this function
    1. Define set of vowels (both upper and lower case)
    2. Iterate through characters
    3. Check if character is alphabetic
    4. Count if it's a vowel or consonant
    """
    vowels = 'ueoaiUEOAI'
    vowels_count = 0 
    consonant_count = 0

    for char in text: 
        if char.isalpha(): 
            if char in vowels: 
                vowels_count += 1
            else: 
                consonant_count += 1 
    return {'vowels': vowels_count, 'consonants': consonant_count}

# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "Hello World 123!"
    print(count_vowels_consonants(text1))  # Expected: {'vowels': 3, 'consonants': 7}
    
    # Test 2
    text2 = "Python Programming"
    print(count_vowels_consonants(text2))  # Expected: {'vowels': 4, 'consonants': 13}
    
    # Test 3
    text3 = "aeiou"
    print(count_vowels_consonants(text3))  # Expected: {'vowels': 5, 'consonants': 0}
    
    # Test 4
    text4 = "xyz"
    print(count_vowels_consonants(text4))  # Expected: {'vowels': 0, 'consonants': 3}
