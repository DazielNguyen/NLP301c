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

def process(text: str) -> list:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    text = "Hello world! How are you today? I am learning NLP."
    print(process(text))
