# Question 2: (2 marks)
# Write a program to extract and print all the nouns present in the below text.
# Hint: using spacy package
# Example:
# Input:
# "James works at Microsoft. She lives in Manchester and likes to play the flute."
# Output:
# James
# Microsoft
# Manchester flute



import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Input text
text = "James works at Microsoft. She lives in Manchester and like to play the flute."
sentences = nltk.sent_tokenize(text)

# Extract and print the nouns
all_nouns = []
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith("NN")]
    all_nouns.extend(nouns)

for noun in all_nouns:
    print(noun)
