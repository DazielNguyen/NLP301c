# Question 16: (2 marks)
# Write a program to reverse the order of words in a sentence,
# but keep each word's characters in their original order.

# Input:
# "Natural Language Processing is amazing"

# Desired Output:
# "amazing is Processing Language Natural"

def reverse_word_order(text: str) -> str:
    # TODO: Implement your solution here
    words = text.split()
    return ' '.join(reversed(words))

if __name__ == '__main__':
    text = "Natural Language Processing is amazing"
    print(reverse_word_order(text))
