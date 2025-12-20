# Question 5: (2 marks)
# Write a program to find the top 3 most frequent words in a text (case-insensitive).
# If there are ties, maintain alphabetical order.

# Input:
# "the quick brown fox jumps over the lazy dog the fox is quick"

# Desired Output:
# [('the', 3), ('fox', 2), ('quick', 2)]

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.lower().split()
    freq = {}
    
    # Đếm frequency
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    
    # Sắp xếp theo frequency giảm dần, nếu bằng nhau thì theo alphabet
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    
    # Lấy top 3
    return sorted_items[:3]

if __name__ == '__main__':
    text = "the quick brown fox jumps over the lazy dog the fox is quick"
    print(process(text))
