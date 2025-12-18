# Question 3: (3 marks)
# Write a program to extract all words longer than 5 characters from a given text. Words are separated by spaces.
# Input: A string containing the text.
# Desired Output: A list of words longer than 5 characters, in the order they appear.
# Example:
# Input: "Natural Language Processing is fascinating"
# Output: ['Natural', Language', 'Processing', 'fascinating']


def process(input):

    words = input.split()

    return [word for word in words if len(word) > 5]

input = 'Natural Language Processing is fascinating'

print(process(input))