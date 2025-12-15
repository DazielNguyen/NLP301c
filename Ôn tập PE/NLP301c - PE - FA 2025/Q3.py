# Question 3: (3 marks)
# Write a program that extracts all words that:
# • contain at least one digit,
# • contain only alphanumeric characters (no punctuation),
# • and return them in the order they appear.
# Input: A string containing the text.
# Desired Output: A list of alphanumeric words containing digits.
# Example:
# Input: "I bought 2apples and 3bananas but item#4 was missing"
# Output: ['2apples', '3bananas']

def extract_words_with_digits(text):
    words = text.split()
    result = []
    
    for word in words:
        if any(char.isdigit() for char in word) and word.isalnum():
            result.append(word)
    
    return result


text = "I bought 2apples and 3bananas but item#4 was missing"
result = extract_words_with_digits(text)
print(result)