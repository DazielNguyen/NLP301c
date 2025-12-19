#Question 2: (2 marks)
# Create a Python program to analyze a given text.
# - Compare two texts to determine their similarity.
# - Use a simple technique like Jaccard similarity to measure similarity.
# Input:
#   text1 = "The quick brown fox jumps over the lazy dog"
#   text2 = "A quick brown fox jumped over a lazy dog"
# Desired Output:
#   Jaccard similarity: 0.5

def jaccard_similarity(text1, text2):
    # Chuyển về lowercase và tách thành set các từ
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Tính intersection (giao) và union (hợp)
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Jaccard similarity = |A ∩ B| / |A ∪ B|
    similarity = len(intersection) / len(union)
    
    return similarity

if __name__ == '__main__':
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A quick brown fox jumped over a lazy dog"
    
    result = jaccard_similarity(text1, text2)
    print(f"Jaccard similarity: {result}")


