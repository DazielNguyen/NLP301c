# Q4: Write a function that takes a list of words (containing duplicates) and 
# returns a list of words (with no duplicates) sorted by ascending frequency.
# Input: input_words = ['table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair']
# Output : ['chair', 'table']

def process(input: list) -> list: 
    freq = {}

    for word in input:
        freq[word] = freq.get(word, 0) + 1

    sorted_word = sorted(freq.items(), key=lambda x: x[1])
    result = [word for word, count in sorted_word]
    
    return result


if __name__ == '__main__': 
    input = ['table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair']
    print(process(input))
    