# Question 1: (2 marks)
# Write a program to tokenize a paragraph into sentences and then tokenize each sentence into words.
# Return a list of lists where each inner list contains the words of a sentence.

# Input:
# "Hello world! How are you today? I am learning NLP."

# Desired Output:
# [['Hello', 'world', '!'], ['How', 'are', 'you', 'today', '?'], ['I', 'am', 'learning', 'NLP', '.']]

# Hints:
# - Use NLTK's sent_tokenize for sentences
# - Use NLTK's word_tokenize for words
# - nltk.download('punkt') if needed

import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

def process(text: str) -> list:
    # TODO: Implement your solution here

    sentences = sent_tokenize(text) # Tách thành các câu nhỏ
    result = [] 
    for sentence in sentences: 
        tokens = word_tokenize(sentence) # Tách các từ trong các câu nhỏ
        result.append(tokens) # Thêm các câu vào list
    return result

if __name__ == '__main__':
    text = "Hello world! How are you today? I am learning NLP."
    print(process(text))
