# Question 5: (2 marks)
# Write a program that reverses each word in a sentence while keeping the word order intact.
# Input: A string containing a sentence.
# Desired Output: A string with each word reversed.
# 
# Example:
# Input: "Hello world from Python"
# Output: "olleH dlrow morf nohtyP"
# 
# Hints:
# - Split the sentence into words
# - Reverse each word using slicing [::-1]
# - Join the reversed words back together

def reverse_words(sentence: str) -> str:
    """
    TODO: Implement this function
    1. Split the sentence into a list of words
    2. Use list comprehension to reverse each word
    3. Join the reversed words with spaces
    """
    words = sentence.split(' ') # Tách thành một câu thành nhiều từ
    reversed = [word[::-1] for word in words] # Đảo từng từ lại
    result = ' '.join(reversed) # Sau đó dùng join ghép lại thành 1 câu
    
    return result

# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "Hello world from Python"
    print(reverse_words(text1))  # Expected: "olleH dlrow morf nohtyP"
    
    # Test 2
    text2 = "Natural Language Processing"
    print(reverse_words(text2))  # Expected: "larutaN egaugnaL gnissecorP"
    
    # Test 3
    text3 = "a b c"
    print(reverse_words(text3))  # Expected: "a b c"
