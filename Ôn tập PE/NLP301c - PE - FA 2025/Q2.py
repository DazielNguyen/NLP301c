# Question 2: (2 marks)
# Write a program that extracts all words longer than 5 characters from a text (case-insensitive) and prints them without duplicates, preserving the order of first appearance.
# Input: A string containing the text.
# Desired Output: A list of unique long words (length > 5).
# Example:
# Input: "machine learning is fun and machine learning improves skills"
# Output: ['machine', 'learning', 'improves', 'skills']

def extract_long_words(text):
    words = text.split()
    seen = set()
    long_words = []
    
    for word in words:
        lower_word = word.lower()
        if len(lower_word) > 5 and lower_word not in seen:
            seen.add(lower_word)
            long_words.append(lower_word)
    
    return long_words

text = "machine learning is fun and machine learning improves skills"
result = extract_long_words(text)
print(result)