# Question 10: (3 marks)
# Write a program to extract all email addresses from a text.
# Return them in lowercase without duplicates.

# Input:
# "Contact us at Support@Company.COM or sales@company.com. For careers, email HR@company.com"

# Desired Output:
# ['support@company.com', 'sales@company.com', 'hr@company.com']

# Hints:
# - Use regex or string matching
# - Email format: username@domain.extension

def process(text: str) -> list:
    # TODO: Implement your solution here
    words = text.split()
    emails = set()
    
    for word in words:
        # Loại bỏ dấu câu cuối
        clean_word = word.strip('.,!?;:')
        
        # Kiểm tra có @ và .
        if '@' in clean_word and '.' in clean_word.split('@')[-1]:
            emails.add(clean_word.lower())
    
    return sorted(list(emails))

if __name__ == '__main__':
    text = "Contact us at Support@Company.COM or sales@company.com. For careers, email HR@company.com"
    print(process(text))
