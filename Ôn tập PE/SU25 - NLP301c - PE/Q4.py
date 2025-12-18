# Question 4: (3 marks)
# Write a program to check if a given string is a valid Python identifier. A valid identifier:
# • Starts with a letter (a-z, A-Z) or an underscore ().
# • Is followed by letters, digits (0-9), or underscores.
# • Is not a Python keyword. (Use the keyword module to check for keywords)
# Input: A string.
# Desired Output: True if it is a valid identifier, False otherwise.
# Example:
# Input: "my variablel"
# Output: True
# Input: "2nd variable"
# Output: False
# Input: "for"
# Output: False



import keyword

def process(input):

    #Check start (a-z, A-Z) or (_)
    start_letter = input[0]

    if keyword.iskeyword(input):
        return False

    if start_letter.isdigit():
        return False
    
    for c in input:
        if not c.isdigit():
            if not c.isalnum():
                if not c == '_':
                    return False

    return True

input_1 = 'my_variable1'
input_2 = '2nd_variable'
input_3 = 'for'

print('Test 1_SP24:', process(input_1))

print('Test 2_SP24_2:', process(input_2))

print('Test 3_SU24_1:', process(input_3))
