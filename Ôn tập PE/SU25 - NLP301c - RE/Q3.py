# Question 3: (3 marks)
# Write a program to extract all email addresses from a given text.
    # Input: A string containing the text.
    # Desired Output: A list of email addresses found in the text.
# Example:
    # Input: "Contact us at support@nlp.com or info@textprocessing.ai for more details."
    # Output: [support@nlp.com, info@textprocessing.ai]


def process(input):

    words = input.split()

    return [word for word in words if word.__contains__('@')]

input = 'Contact us at support@nlp.com or info@textprocessing.ai for more details.'

print(process(input))