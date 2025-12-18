# Question 1: (2 marks)
# Write a program to reverse the characters in each word of a sentence, while keeping the word order the same.
    # Input: A string containing the sentence.
    # Desired Output: The sentence with each word's characters reversed.
# Example:
    # Input: "Natural Language Processing"
    # Output: "larutaN egaugnal gnissecorP"


def process(input):

    ans = ''

    words = input.split()
    for word in words:
        l = len(word)

        tmp = ''

        for i in range(l-1,-1,-1):
            tmp += word[i] # word[i] = character
        
        if ans == '':
            ans = tmp
        else:
            ans = ans + ' ' + tmp
    
    return ans

input = 'Natural Language Processing'

print(process(input))
