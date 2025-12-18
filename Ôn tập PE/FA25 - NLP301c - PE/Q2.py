# Question 2: (2 marks)
# Write a program that extracts all words longer than 5 characters from a text (case-insensitive) and 
# prints them without duplicates, preserving the order of first appearance.
# Input: A string containing the text.
# Desired Output: A list of unique long words (length > 5).
# Example:
# Input: "machine learning is fun and machine learning improves skills"
# Output: ['machine', 'learning', 'improves', 'skills']


def process(sentence) -> list: 
    words = sentence.split(' ')
    seen = set()
    result = []

    for word in words: 
        words_lower = word.lower()
        if len(words_lower) > 5 and words_lower not in seen: 
            seen.add(words_lower)
            result.append(words_lower)
    return result

if __name__ == '__main__': 
    text = "machine learning is fun and machine learning improves skills"
    print(process(text))
