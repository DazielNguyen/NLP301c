# Question 15: (2 marks)
# Write a program to convert text to title case while preserving acronyms.
# Acronyms (all caps words) should remain uppercase.

# Input:
# "hello from NASA and FBI headquarters in USA"

# Desired Output:
# "Hello From NASA And FBI Headquarters In USA"

# Hints:
# - Check if word is already all uppercase
# - If yes, keep it; otherwise, capitalize

def smart_title_case(text: str) -> str:
    # TODO: Implement your solution here
    words = text.split()
    result = []
    
    for word in words:
        # Kiểm tra xem có phải tất cả chữ hoa không (acronym)
        if word.isupper():
            result.append(word)
        else:
            result.append(word.capitalize())
    
    return ' '.join(result)

if __name__ == '__main__':
    text = "hello from NASA and FBI headquarters in USA"
    print(smart_title_case(text))
