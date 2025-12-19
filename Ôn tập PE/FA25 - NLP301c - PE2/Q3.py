# Question 3: (3 marks)
# Write a program to extract all words that start with a consonant and contain at least one repeated letter.
    # Input: A string containing the text.
    # Desired Output: A list of qualifying words in the order they appear.
# Example:
    # Input: "better apple assignment committee cool loop"
    # Output: ['better", 'committee', 'cool", loop]


def process(sentence: str) -> list: 
    words = sentence.split()
    vowels = 'ueoaiUEOAI'
    result = []

    for word in words: 
        if word[0] not in vowels: 
            if len(word) != len(set(word)):
                result.append(word)
    return result
if __name__ == '__main__': 
    text = "better apple assignment committee cool loop"
    print(process(text))

