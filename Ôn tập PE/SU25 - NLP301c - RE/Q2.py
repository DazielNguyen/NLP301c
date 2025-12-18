# Question 2: (2 marks)
# Write a Python program to find the most common bigram (pair of consecutive words) in a given text.
    # Input: A string containing the text.
    # Desired Output: The most common bigram as a tuple.
# Example:
    # Input: "Deep learning is great. Deep learning is the future." 
    # Output: "Most common bigram: (Deep', '"learning)"


def process(input):
    
    list_count = {} # dictionary has key/value

    words = input.split()

    l = len(words)

    for i in range(1, l-1):
        pair = (words[i - 1], words[i]) # () la tuple

        if (pair in list_count):
            list_count[pair] += 1
        else:
            list_count[pair] = 1

    max_value = 0
    ans = ''
    for key, value in list_count.items():
        if max_value < value:
            max_value = value
            ans = key

    return ans

input = 'Deep learning is great. Deep learning is the future.'

print(f'Most common bigram: {process(input)}')