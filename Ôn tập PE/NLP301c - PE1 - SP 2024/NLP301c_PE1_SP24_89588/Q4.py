def sort_by_frequency(words):
    # Create a frequency dictionary
    freq_dict = {}
    for word in words:
        freq_dict[word] = freq_dict.get(word, 0) + 1
    
    # Sort the dictionary by frequency (ascending)
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1])
    
    # Extract the unique words from the sorted list
    unique_words = [word for word, freq in sorted_freq]
    
    return unique_words

input_words = ['table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'table', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair']
result = sort_by_frequency(input_words)

# 10 table and 9 chair so output will be ['chair', 'table']
print(result)