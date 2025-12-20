# Question 4: (2 marks)
# Write a program to count the frequency of each word (case-insensitive) and 
# return only words that appear more than once, sorted by frequency in descending order.

# Input:
# "Apple banana apple Cherry banana apple orange Banana"

# Desired Output:
# [('apple', 3), ('banana', 3)]

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.lower().split()
    result = []
    freq = {}
    # Đếm từ
    for word in words: 
        if word in freq:
            freq[word] += 1

        else: 
            freq[word] = 1
    
    filtered = [(word, count) for word, count in freq.items() if count > 1]
    result = sorted(filtered, key=lambda x: x[1] ,reverse=True)
    return result
        

if __name__ == '__main__':
    text = "Apple banana apple Cherry banana apple orange Banana"
    print(process(text))
