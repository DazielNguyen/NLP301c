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
    words = text.lower().split()
    result = []
    seen = set()
    stop_words_set = set(word.lower() for word in stop_words)
    
    for word in words:
        if word not in stop_words_set and word not in seen:
            result.append(word)
            seen.add(word)
    
    return result

if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog and the fox runs fast"
    stop_words = ["the", "a", "an", "and", "or", "but"]
    print(process(text, stop_words))
