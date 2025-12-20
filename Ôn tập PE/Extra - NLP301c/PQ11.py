# Question 11: (3 marks)
# Write a program to find all palindrome words in a text (case-insensitive).
# A palindrome reads the same forwards and backwards.
# Return unique palindromes in lowercase, sorted alphabetically.

# Input:
# "Mom and Dad saw a racecar at noon. The kayak was level on the water."

# Desired Output:
# ['dad', 'kayak', 'level', 'mom', 'noon', 'racecar']

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.split()
    palindromes = set()
    
    for word in words:
        # Loại bỏ dấu câu
        clean_word = ''
        for char in word:
            if char.isalpha():
                clean_word += char
        
        if clean_word:
            word_lower = clean_word.lower()
            # Kiểm tra palindrome
            if word_lower == word_lower[::-1]:
                palindromes.add(word_lower)
    
    return sorted(list(palindromes))

if __name__ == '__main__':
    text = "Mom and Dad saw a racecar at noon. The kayak was level on the water."
    print(process(text))
