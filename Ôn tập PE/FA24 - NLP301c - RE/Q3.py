# Question 3: (3 marks)
# Write a program to extract the Verb Phrases from the given text.
# Input:
    # I may bake a cake for my birthday. The talk will introduce reader about Use of baking.
# Desired Output:
    # ['may bake', 'will introduce']

import nltk
from nltk import pos_tag, word_tokenize

def process(sentence: str) -> list: 
    # Tokenize và POS tagging
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    # Pattern: Modal (MD) + Verb (VB)
    # MD: modal verb (may, will, can, should...)
    # VB: base form verb (bake, introduce...)
    grammar = "VP: {<MD><VB>}" # Thay đổi nếu muốn lấy NP "NP: {<DT>?<JJ>*<NN.*>+}"

    # Tạo chunk parser
    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(tagged)
    
    # Trích xuất verb phrases
    verb_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'VP':
            # Lấy các từ trong VP
            vp = ' '.join(word for word, tag in subtree.leaves())
            verb_phrases.append(vp)
    
    return verb_phrases

if __name__ == '__main__': 
    text = "I may bake a cake for my birthday. The talk will introduce reader about Use of baking."
    result = process(text)
    print(result)


  
    # Pattern cho Noun Phrase:
    # DT: determiner (the, a, an)
    # JJ: adjective (quick, brown)
    # NN: noun (dog, cat, cake)
