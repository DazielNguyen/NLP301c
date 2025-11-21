def count_vowels(word):
    vowels = "aeiou"
    count = 0
    for char in word.lower():
        if char in vowels:
            count += 1
    return count


def process_words(word_list):
    word_vowels = [[] for _ in range(max(len(word) for word in word_list) + 1)]
    print("Word length | Number of vowels")
    for word in word_list:
        l = len(word)
        v = count_vowels(word)
        if not word_vowels[l]:
            word_vowels[l] = [
                set()
                for _ in range(
                    max(count_vowels(w) for w in word_list if len(w) == l) + 1
                )
            ]
        word_vowels[l][v].add(word)
        # Print the word and the number of vowels of the word
        print(f"{l}    {v}")


word_list = "Write code to initialize an array and process a list of words".split()
process_words(word_list)
