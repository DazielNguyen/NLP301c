# Question 4: (3 marks)
# Write a program to check whether a given sentence is 
# an isogram - a word or sentence where no letter is repeated (ignoring case and non-alphabetic characters).
# Input: A string containing the sentence.
# Desired Output: True or False
# Example:
# Input: "learning"
# Output: False
# Input: "lamp"
# Output: True

def is_isogram(sentence):
    cleaned_sentence = ''.join(char.lower() for char in sentence if char.isalpha())
    seen = set()
    
    for char in cleaned_sentence:
        if char in seen:
            return False
        seen.add(char)
    
    return True

sentence1 = "learning"
sentence2 = "lamp"
result1 = is_isogram(sentence1)
result2 = is_isogram(sentence2)
print(result1)  # Output: False
print(result2)  # Output: True