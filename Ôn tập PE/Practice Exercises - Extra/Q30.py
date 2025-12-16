# Question 30: (4 marks)
# Write a program that creates an inverted index from a list of documents.
# An inverted index maps each word to the list of document IDs where it appears.
# Input: A list of strings (documents).
# Desired Output: A dictionary where keys are words and values are lists of document indices.
# 
# Example:
# Input: [
#     "the quick brown fox",
#     "the lazy dog",
#     "the quick dog"
# ]
# Output: {
#     'the': [0, 1, 2],
#     'quick': [0, 2],
#     'brown': [0],
#     'fox': [0],
#     'lazy': [1],
#     'dog': [1, 2]
# }
# 
# Hints:
# - Iterate through documents with enumerate
# - Split each document into words
# - For each word, track document indices
# - Use set to avoid duplicate indices

def create_inverted_index(documents):
    """
    TODO: Implement this function
    1. Create empty dictionary for index
    2. Iterate through documents with enumerate
    3. For each word in document, add document index
    4. Return sorted inverted index
    """
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    docs1 = [
        "the quick brown fox",
        "the lazy dog",
        "the quick dog"
    ]
    result1 = create_inverted_index(docs1)
    print(result1)
    # Expected: {'the': [0, 1, 2], 'quick': [0, 2], 'brown': [0], 
    #            'fox': [0], 'lazy': [1], 'dog': [1, 2]}
    
    # Test 2
    docs2 = [
        "hello world",
        "hello python",
        "python world"
    ]
    result2 = create_inverted_index(docs2)
    print(result2)
    # Expected: {'hello': [0, 1], 'world': [0, 2], 'python': [1, 2]}
    
    # Test 3
    docs3 = [
        "apple apple",
        "banana",
        "apple banana"
    ]
    result3 = create_inverted_index(docs3)
    print(result3)
    # Expected: {'apple': [0, 2], 'banana': [1, 2]}
