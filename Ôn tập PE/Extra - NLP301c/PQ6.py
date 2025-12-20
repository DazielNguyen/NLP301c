# Question 6: (2 marks)
# Write a program to calculate the average length of words in a text, 
# excluding words shorter than 3 characters. Round to 2 decimal places.

# Input:
# "I am learning natural language processing with Python"

# Desired Output:
# 8.17

# Explanation: Words >= 3 chars: ["learning", "natural", "language", "processing", "with", "Python"]
# Average length: (8+7+8+10+4+6)/6 = 7.17

def process(text: str) -> float:
    # TODO: Implement your solution here
    words = text.split()
    
    # Lọc từ có độ dài >= 3
    filtered_words = [word for word in words if len(word) >= 3]
    
    # Tính trung bình
    if len(filtered_words) == 0:
        return 0.0
    
    total_length = sum(len(word) for word in filtered_words)
    average = total_length / len(filtered_words)
    
    return round(average, 2)

if __name__ == '__main__':
    text = "I am learning natural language processing with Python"
    print(process(text))
