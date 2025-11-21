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
