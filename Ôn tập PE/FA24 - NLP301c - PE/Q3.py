# Question 3: (3 marks)
# Write a program to correct the spelling errors in the given text

# Input:
        # He is a gret person. He beleives in bod
# Desired Output:
        # He is a great person. He believes in god
# Nhớ tải thư viện
# pip install textblob
# python3 -m textblob.download_corpora

from textblob import TextBlob

def correct_spelling(text):
    # Tạo TextBlob object
    blob = TextBlob(text)
    
    # Sửa lỗi chính tả
    corrected = blob.correct()
    
    return str(corrected)

if __name__ == '__main__':
    text = "He is a gret person. He beleives in bod"
    result = correct_spelling(text)
    print(result)