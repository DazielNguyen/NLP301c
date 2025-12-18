# Question 4: (3 marks)
# Write code to initialize a two-dimensional array of sets called word_ vowels and process a list of words,
# adding each word to word vowels[1][v] where 1 is the length of the word and v is the number of vowels it contains. Print 1 and v
# Example:
# Input:
# word_list = Write code to initialize an array and process a list of words'
# Output:
# 5
# 1

def vowels(word):
    v = ['a','e','i','o','u']

    dem = 0
    for i in v:
        if i in word:
            dem += 1

    return dem

def process(word_list):

    words = word_list.split()

    for i in range(len(words)):
        if words[i] == 'an':
            return words[i+1]

word_list = "Write code to initialize an array and process a list of words"

print(len(process(word_list)))
print(vowels(process(word_list)))