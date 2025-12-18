# Question 4: (3 marks)
# Write a Python program that takes a sentence as input and outputs a new sentence where all
# the words are sorted alphabetically.
# Input:
# A sentence: "the quick brown fox jumps over the lazy dog"
# Output:
# "brown dog fox jumps lazy over quick the the"






def sort_sentence(sentence):
    words = sentence.split()
    sorted_words = sorted(words)
    sorted_sentence = " ".join(sorted_words)
    return sorted_sentence

def main() -> None:
    sentence = "the quick brown fox jumps over the lazy dog"
    output = sort_sentence(sentence)
    print(output)  # Output: "brown dog fox jumps lazy over quick the the"

if __name__ == "__main__":
    main()

