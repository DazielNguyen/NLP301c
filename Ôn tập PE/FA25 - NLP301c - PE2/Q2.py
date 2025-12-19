# Question 2: (2 marks)
# Write a program to count how many words in the given text contain at least one vowel (a, e, i, o, u ignore case).
    # Input: A string containing the text.
    # Desired Output: An integer representing the number of words containing vowels.
# Example:
    # Input: "Sky rhythm fly by"
    # Output: 0
    # Input: "Natural language fly"
    # Output: 2



def process(sentence: str) -> int:
    words = sentence.split()
    vowels = ['u', 'e', 'o', 'a', 'i']

    count_vowels = 0
    
    for word in words:
        # Chuyển từ về chữ thường để so sánh
        word_lower = word.lower()
        
        # Kiểm tra xem từ có chứa ít nhất 1 nguyên âm không
        has_vowel = False
        for char in word_lower:
            if char in vowels:
                has_vowel = True
                break
        
        # Nếu có nguyên âm thì tăng count
        if has_vowel:
            count_vowels += 1
    return count_vowels

if __name__ == '__main__': 
    text1 = "Sky rhythm fly by"
    text2 = "Natural language fly"
    
    print(process(text1))  # Kết quả: 0
    print(process(text2))  # Kết quả: 2
