def sentence_to_word_list(sentence: str) -> list[str]:
    # Remove leading and trailing whitespace, then split the sentence into words
    words = sentence.strip().split()
    
    # Remove any punctuation from the words
    cleaned_words = [word.strip(".,!?\"'") for word in words]
    
    return cleaned_words

def main() -> None:
    # Hardcoded input sentence
    input_sentence = "The quick brown fox jumps over the lazy dog"

    # Process the sentence and get the list of words
    word_list = sentence_to_word_list(input_sentence)

    # Display the result
    print("Input sentence:")
    print(input_sentence)
    print("\nList of words:")
    print(word_list)

if __name__ == "__main__":
    main()