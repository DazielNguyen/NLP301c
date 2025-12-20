# Question 3: (2 marks)
# Write a program to remove all punctuation marks from a text and convert it to lowercase.

# Input:
# "Hello, World! This is a Test... How are you?"

# Desired Output:
# "hello world this is a test how are you"

# Hints:
# - Use string.punctuation
# - Use str.translate() and str.maketrans()


# Solution 1: Không dùng thư viện String

def process(text: str) -> str:
    # TODO: Implement your solution here
    punctuation = "'!`~@#$%^&*()_-+=[|]{\}.,<>?/""'''"
    result = []
    words = text.lower().split()
    for word in words: 
        clean_word = ""
        for char in word:
            if char not in punctuation: 
                clean_word += char
        if clean_word:
            result.append(clean_word)
    return ' '.join(result)


# Solution 2: Dùng thư viện String
# import string
# def process_v2(text: str) -> str:
#     translator = str.maketrans('', '', string.punctuation)
#     return text.lower().translate(translator)


if __name__ == '__main__':
    text = "Hello, World! This is a Test... How are you?"
    print(process(text))
