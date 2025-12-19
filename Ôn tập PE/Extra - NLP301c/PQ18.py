# Question 18: (3 marks)
# Write a program to group words by their length and return a dictionary
# where keys are lengths and values are lists of unique words (lowercase) of that length,
# sorted alphabetically.

# Input:
# "The cat and the dog ran to the big red car"

# Desired Output:
# {2: ['to'], 3: ['and', 'big', 'cat', 'dog', 'ran', 'red', 'the'], 3: ['car']}

def group_by_length(text: str) -> dict:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "The cat and the dog ran to the big red car"
    print(group_by_length(text))
