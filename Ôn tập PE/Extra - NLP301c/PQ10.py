# Question 10: (3 marks)
# Write a program to extract all email addresses from a text.
# Return them in lowercase without duplicates.

# Input:
# "Contact us at Support@Company.COM or sales@company.com. For careers, email HR@company.com"

# Desired Output:
# ['support@company.com', 'sales@company.com', 'hr@company.com']

# Hints:
# - Use regex or string matching
# - Email format: username@domain.extension

def process(text: str) -> list:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "Contact us at Support@Company.COM or sales@company.com. For careers, email HR@company.com"
    print(process(text))
