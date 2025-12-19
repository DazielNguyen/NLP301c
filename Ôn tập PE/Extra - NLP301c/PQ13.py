# Question 13: (3 marks)
# Write a program to validate Python variable names.
# A valid Python variable name must:
# 1. Start with a letter (a-z, A-Z) or underscore (_)
# 2. Contain only letters, digits, or underscores
# 3. Not be a Python keyword

# Input:
# ["my_var", "2nd_var", "_private", "for", "myVar123", "my-var", "class"]

# Desired Output:
# {'my_var': True, '2nd_var': False, '_private': True, 'for': False, 'myVar123': True, 'my-var': False, 'class': False}

def validate_variables(names: list) -> dict:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    names = ["my_var", "2nd_var", "_private", "for", "myVar123", "my-var", "class"]
    print(validate_variables(names))
