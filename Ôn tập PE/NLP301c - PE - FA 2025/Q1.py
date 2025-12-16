# Question 1: (2 marks)
# Write a program that counts the frequency of each word in a given text (ignoring case) 
# and prints the result as a dictionary sorted alphabetically by word.
# Input: A string containing the text.
# Desired Output: A dictionary where
# • keys= lowercase words
# • values = counts
# • dictionary sorted alphabetically by keys
# Example:
# Input: "Cat dog cat Bird DOG bird dog"
# Output: {'bird': 2, 'cat': 2, 'dog': 3}

def process(sentence)-> dict:
    words = sentence.lower().split(' ')
    freq = {}
    for word in words: 
        if word in freq: 
            freq[word] += 1
        else: 
            freq[word] = 1

    result = dict(sorted(freq.items()))
    
    return result
if __name__ == '__main__': 
    text = "Cat dog cat Bird DOG bird dog"
    print(process(text))






















