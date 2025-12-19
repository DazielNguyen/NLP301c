# Question 4: (3 marks)
# Write a program to create bigrams from the given texts using Gensim library's Phrases

# Input:
    # The mayor of new york was there", "new york mayor was present
# Desired Output:
    # ['The', 'mayor', 'of, 'new_york', 'was', 'there'] 
    # ['new_york', 'mayor', 'was', 'present]

from gensim.models import Phrases
from gensim.models.phrases import Phraser

def create_bigrams(texts):
    # Tokenize các câu thành list of words
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Tạo bigram model với Phrases
    # min_count: số lần xuất hiện tối thiểu
    # threshold: ngưỡng để tạo bigram (càng thấp càng dễ tạo bigram)
    bigram_model = Phrases(tokenized_texts, min_count=1, threshold=1)
    
    # Tạo Phraser để xử lý nhanh hơn
    bigram_phraser = Phraser(bigram_model)
    
    # Áp dụng bigram cho từng text
    result = []
    for tokens in tokenized_texts:
        bigram_tokens = bigram_phraser[tokens]
        result.append(list(bigram_tokens))
    
    return result

if __name__ == '__main__':
    texts = [
        "The mayor of new york was there",
        "new york mayor was present"
    ]
    
    result = create_bigrams(texts)
    for bigrams in result:
        print(bigrams)