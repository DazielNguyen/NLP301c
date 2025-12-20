# Question 8: (3 marks)
# Write a program to extract all words that contain both a vowel and a digit.
# Words should only contain alphanumeric characters (no special characters).

# Input:
# "user123 has admin456 access but item#789 was blocked. test7abc works!"

# Desired Output:
# ['user123', 'admin456', 'test7abc']

# Explanation: 
# - user123: has vowels (u,e) and digits (1,2,3) ✓
# - admin456: has vowels (a,i) and digits (4,5,6) ✓
# - item#789: has # so not alphanumeric ✗
# - test7abc: has vowels (e,a) and digit (7) ✓

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.split()
    vowels = 'aeiouAEIOU'
    result = []
    
    for word in words:
        # Kiểm tra chỉ chứa alphanumeric
        if not word.isalnum():
            continue
        
        # Kiểm tra có vowel và digit
        has_vowel = False
        has_digit = False
        
        for char in word:
            if char in vowels:
                has_vowel = True
            if char.isdigit():
                has_digit = True
        
        if has_vowel and has_digit:
            result.append(word)
    
    return result

if __name__ == '__main__':
    text = "user123 has admin456 access but item#789 was blocked. test7abc works!"
    print(process(text))
