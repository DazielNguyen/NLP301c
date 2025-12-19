# Question 2: (2 marks)
# Create a Python program to analyze a given text.
# - Count the frequency of each word in a given text.
# - Sort the words by their frequency in descending order.

# Input:
            # This is a sample text. This text is for testing purposes.
# Desired Output:
            # This: 2
            # is: 2
            # a: 1
            # sample: 1
            # text.: 1
            # text: 1
            # for: 1
            # testing: 1
            # purposes.: 1
# import string
def process(sentence) -> dict: 
    # Muốn bỏ dấu câu trong câu thì thêm dòng này, nếu như đề yêu cầu
    # sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    words = sentence.split()
    words_count = {}
    for word in words: 
        if word in words_count: 
            words_count[word] += 1
        else: 
            words_count[word] = 1 

    sorted_word = sorted(words_count.items(), key=lambda x: x[1], reverse=True)

    for word, count in sorted_word: 
        print(f"{word}: {count}")

if __name__ == '__main__': 
    text = " This is a sample text. This text is for testing purposes."
    process(text)
