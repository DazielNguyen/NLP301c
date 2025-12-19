# Question 19: (3 marks)
# Write a program to sort words by multiple criteria:
# 1. First by length (ascending)
# 2. Then alphabetically (ascending)
# 3. Case-insensitive
# Return unique words only.

# Input:
# "Zebra apple Cat dog Elephant bee cat APPLE"

# Desired Output:
# ['bee', 'cat', 'dog', 'apple', 'zebra', 'elephant']

# Explanation:
# Length 3: bee, cat, dog (alphabetically)
# Length 5: apple, zebra (alphabetically)
# Length 8: elephant

def multi_sort(text: str) -> list:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "Zebra apple Cat dog Elephant bee cat APPLE"
    print(multi_sort(text))
