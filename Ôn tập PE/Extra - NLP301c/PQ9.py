# Question 9: (2 marks)
# Write a program to extract all unique words from a text (case-insensitive),
# maintaining the order of first appearance, but exclude common stop words.

# Input:
# text = "The quick brown fox jumps over the lazy dog and the fox runs fast"
# stop_words = ["the", "a", "an", "and", "or", "but"]

# Desired Output:
# ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'runs', 'fast']

def process(text: str, stop_words: list) -> list:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog and the fox runs fast"
    stop_words = ["the", "a", "an", "and", "or", "but"]
    print(process(text, stop_words))
