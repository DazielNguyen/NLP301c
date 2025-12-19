# Question 1: (2 marks)

# Given a sentence, write a Python program to tokenize it into words and punctuation marks.

# Input:
        # This is a sample sentence. It's a simple task.
# Desired Output:
        # ['This', 'is', 'a', 'sample', 'sentence', '.' , 'It',"s", 'a', 'simple', 'task', '.']

import nltk
nltk.download('punkt_tab') 
from nltk.tokenize import word_tokenize 

def process(sentence: str) -> list:    
    return word_tokenize(sentence)

if __name__ == '__main__':
    text = "This is a sample sentence. It's a simple task."
    print(process(text))



