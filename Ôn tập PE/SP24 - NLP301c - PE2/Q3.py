# Question 3: (3 marks)

# Write a function that takes a text and a vocabulary as its arguments and 
# returns the set of words that appear in the text and in the vocabulary. 
# Both arguments can be represented as lists of strings.
# Example:
# Input:
# text = 'a text and a vocabulary'
# vocab = 'a vocabulary'
# Output:
# ['a', 'vocabulary']





def find_common_words(text, vocabulary):
    if isinstance(text, str):
        text_words = set(text.split())
    else:
        text_words = set(word for line in text for word in line.split())
    
    if isinstance(vocabulary, str):
        vocabulary = vocabulary.split()
    else:
        vocabulary = [word for line in vocabulary for word in line.split()]

    vocab_set = set(vocabulary)
    common_words = list(text_words.intersection(vocab_set))
    return common_words


text = "a text and a vocabulary world"
vocab = "a vocabulary"
common_words = find_common_words(text, vocab)
print(common_words)
