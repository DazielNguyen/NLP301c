# Question 6: (2 marks)
# Write a program to calculate the average length of words in a text, 
# excluding words shorter than 3 characters. Round to 2 decimal places.

# Input:
# "I am learning natural language processing with Python"

# Desired Output:
# 8.17

# Explanation: Words >= 3 chars: ["learning", "natural", "language", "processing", "with", "Python"]
# Average length: (8+7+8+10+4+6)/6 = 7.17

def process(text: str) -> float:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "I am learning natural language processing with Python"
    print(process(text))
