# Question 1: (2 marks)
# Write a program to convert a given text to title case and remove extra spaces. Assume the text contains only letters and spaces.
# Input: A string containing the text.
# Desired Output: The text in title case with no extra spaces.
# Example:
# Input: "hello world this is a test"
# Output: "Hello World This Is A Test"


import re

def process(input):

    words = input.split()

    a = [word.capitalize() for word in words]

    ans = ''
    for word in a:
        if ans == '':
            ans = word
        else:
            ans = ans + ' ' + word
    
    return ans

input = '  hello    world     this    is   a   test   '

print(process(input))