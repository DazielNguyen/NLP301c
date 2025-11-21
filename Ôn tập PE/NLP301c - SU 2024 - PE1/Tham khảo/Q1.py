def sentence_to_word_list(sentence: str) -> list[str]:
    words = sentence.strip().split()
    
    cleaned_words = [word.strip(".,!?\"'") for word in words]
    
    return cleaned_words

def main() -> None:
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