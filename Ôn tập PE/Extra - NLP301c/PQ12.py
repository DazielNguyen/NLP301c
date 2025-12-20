# Question 12: (3 marks)
# Write a program to check if a given text is an isogram.
# An isogram is a word/phrase where no letter appears more than once.
# Ignore spaces, punctuation, and case.

# Input 1: "Dermatoglyphics"
# Output 1: True

# Input 2: "Programming"
# Output 2: False (g, m, r appear twice)

# Input 3: "The quick brown fox"
# Output 3: False (o appears twice)

def is_isogram(text: str) -> bool:
    # TODO: Implement your solution here
    # Chỉ lấy chữ cái, chuyển thành lowercase
    letters = ''
    for char in text:
        if char.isalpha():
            letters += char.lower()
    
    # Kiểm tra xem có chữ nào lặp lại không
    seen = set()
    for letter in letters:
        if letter in seen:
            return False
        seen.add(letter)
    
    return True

if __name__ == '__main__':
    test1 = "Dermatoglyphics"
    test2 = "Programming"
    test3 = "The quick brown fox"
    
    print(f"'{test1}': {is_isogram(test1)}")
    print(f"'{test2}': {is_isogram(test2)}")
    print(f"'{test3}': {is_isogram(test3)}")
