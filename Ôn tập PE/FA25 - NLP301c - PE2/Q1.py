# #Question 1: (2 marks)
# Write a program to remove all punctuation from a given text and convert the remaining text into
# lowercase tokens.

# Input: A string containing the text.
# Desired Output: A list of lowercase tokens without punctuation.

# Example:
# Input: "Hello, world! NLP—is fun."
# Output: ['hello', 'world', 'nlp', 'is', 'fun']


def process(sentence: str) -> list: 
    cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in sentence)
    words = [word for word in cleaned.lower().split() if word]
    return words


if __name__ == '__main__': 
    text = "Hello, world! NLP—is fun."
    print(process(text))