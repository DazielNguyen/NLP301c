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
    words = text.lower().split()
    groups = {}
    
    for word in words:
        length = len(word)
        if length not in groups:
            groups[length] = set()
        groups[length].add(word)
    
    # Chuyển set thành list đã sắp xếp
    result = {}
    for length, word_set in groups.items():
        result[length] = sorted(list(word_set))
    
    return result

if __name__ == '__main__':
    text = "The cat and the dog ran to the big red car"
    print(group_by_length(text))
