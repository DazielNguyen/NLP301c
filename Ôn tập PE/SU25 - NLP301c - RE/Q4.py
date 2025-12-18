# Question 4: (3 marks)
# Write a program to sort sentences in a paragraph by the number of words they contain, from shortest to longest.
    # Input: A string containing the paragraph.
    # Desired Output: A list of sentences sorted by the number of words.
# Example:
    # Input: "The quick brown fox jumps over the lazy dog. It was a sunny day. Birds were chirping."
    # Output: ['It was a sunny day.', 'Birds were chirping.', 'The quick brown fox jumps over the lazy dog.']

def process(input):

    sentences = []

    words = input.split()
    
    tmp = ''

    for word in words:
        if word.endswith('.'):
            tmp = tmp + ' ' + word
            sentences.append(tmp)
            tmp = ''
        else:
            if tmp == '':
                tmp = word
            else:
                tmp = tmp + ' ' + word

    ans = sorted(sentences, key=lambda sentences: len(sentences))
    return ans

input = 'The quick brown fox jumps over the lazy dog. It was a sunny day. Birds were chirping.'

print(process(input))