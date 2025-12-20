# Question 7: (3 marks)
# Write a program to extract all words that:
# 1. Start with a vowel (a, e, i, o, u - case insensitive)
# 2. Have length greater than 3
# 3. Return them in lowercase without duplicates, in order of first appearance

# Input:
# "An elephant is an intelligent animal. Every elephant eats plants."

# Desired Output:
# ['elephant', 'intelligent', 'animal', 'every', 'eats']

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.split()
    vowels = 'aeiouAEIOU'
    result = []
    seen = set()
    
    for word in words:
        # Loại bỏ dấu câu
        clean_word = ''
        for char in word:
            if char.isalpha():
                clean_word += char
        
        # Kiểm tra điều kiện
        if clean_word and len(clean_word) > 3 and clean_word[0] in vowels:
            word_lower = clean_word.lower()
            if word_lower not in seen:
                result.append(word_lower)
                seen.add(word_lower)
    
    return result

if __name__ == '__main__':
    text = "An elephant is an intelligent animal. Every elephant eats plants."
    print(process(text))
