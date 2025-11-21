def sort_sentence(sentence):
    words = sentence.split()
    sorted_words = sorted(words)
    sorted_sentence = " ".join(sorted_words)
    return sorted_sentence

def main() -> None:
    # Hardcoded input sentence
    input_sentence = "The quick brown fox jumps over the lazy dog"

    # Example usage
    sentence = "the quick brown fox jumps over the lazy dog"
    output = sort_sentence(sentence)
    print(output)  # Output: "brown dog fox jumps lazy over quick the the"

if __name__ == "__main__":
    main()

