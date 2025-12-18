# Question 3: # points
# Write a Python program that takes a document as input and outputs a list of the named
# entities (people, organizations, locations) in the document.
# Hint: This requires using a named entity recognition library like spaCy.

# Example:

# Input:
# A document: "Apple Inc. is an American multinational technology company. Its
# headquarters are in Cupertino, California."

# Output:
# [l'Apple Inc.', 'ORG'), ('American', 'NORP'), ('Cupertino', 'GPE'), ('California', 'GPE')]




import sys
import subprocess

try:
    import spacy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

try:
    spacy.load("en_core_web_sm")
except Exception:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def extract_named_entities(document: str) -> list[tuple[str, str]]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(document)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

def main() -> None:
    document = "Apple Inc. is an American multinational technology company. Its headquarters are in Cupertino, California."

    # Extract named entities
    named_entities = extract_named_entities(document)

    # Display the result
    print("Input document:")
    print(document)
    print("\nNamed Entities:")
    print(named_entities)

if __name__ == "__main__":
    main()
