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
