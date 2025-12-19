#Question 1: (2 marks)

# Given a text, write a Python program to stem the words using the PorterStemmer.
# Input:
    # The cats were playing in the garden, and they were having fun.
# Desired Output:
    # ['the', 'cat", 'were", 'play", 'in', the, 'garden', ',' ,'and', 'they', 'were', 'have, 'fun', '.']

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stem_text(text):
    # Khởi tạo PorterStemmer
    stemmer = PorterStemmer()
    
    # Tokenize text thành các từ (giữ cả dấu câu)
    tokens = word_tokenize(text)
    
    # Stem từng token và chuyển về lowercase
    stemmed_words = [stemmer.stem(token.lower()) for token in tokens]
    
    return stemmed_words

if __name__ == '__main__':
    text = "The cats were playing in the garden, and they were having fun."
    result = stem_text(text)
    print(result)

