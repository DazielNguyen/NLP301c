# Question 4: (3 marks)
# Write a program to clean the following tweet and tokenize them

# Input:
    # Having lots of fun #goa #vaction #summervacation. Fancy dinner @Beachbay restro:
# Desired Output:
    # ['Having', lots', 'of', 'fun', 'goa', 'vaction', 'summervacation', 'Fancy', 'dinner', 'Beachbay', 'restro']

def process(sentence): 
    # Loại bỏ các ký tự đặc biệt, chỉ giữ chữ cái, số và khoảng trắng
    cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in sentence)
    # Tokenize và loại bỏ chuỗi rỗng
    words = [word for word in cleaned.split() if word]
    return words

if __name__ == '__main__':
    text = "Having lots of fun #goa #vaction #summervacation. Fancy dinner @Beachbay restro:"
    print(process(text))