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

def word_frequency(text):
    words = text.lower().split()
    freq = {}
    
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    
    sorted_freq = dict(sorted(freq.items()))
    return sorted_freq

text = "Cat dog cat Bird DOG bird dog"
result = word_frequency(text)
print(result)

