# Question 2: (2 marks)
# Write a program to calculate the average length of words in a given text. Words are separated by spaces. Round the result to two decimal places.
# Input: A string containing the text.
# Desired Output: The average word length as a float rounded to two decimal places.
# Example:
# Input: "The quick brown fox"
# Output: 4.00

def process(input):

    words = input.split()

    return len(words)

input = 'The quick brown nigga'

print(f'{process(input):.4f}')