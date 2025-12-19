# Question 17: (3 marks)
# Write a program to extract every nth word from a text, starting from position m.

# Input:
# text = "The quick brown fox jumps over the lazy dog near the river"
# n = 3 (every 3rd word)
# m = 1 (start from position 1, 0-indexed)

# Desired Output:
# ['quick', 'jumps', 'lazy', 'river']

# Explanation: Starting at index 1 (quick), take every 3rd word
# Positions: 1(quick), 4(jumps), 7(lazy), 10(river)

def extract_nth_words(text: str, n: int, m: int) -> list:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog near the river"
    n = 3
    m = 1
    print(extract_nth_words(text, n, m))
