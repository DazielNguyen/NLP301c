# Question 4: (3 marks)
# Write a program that determines whether a given sentence is a tautogram, 
# meaning all significant words start with the same letter 
# (ignore case and ignore stop words: "a", "an", "the", "and").


    # Input: A sentence string.
    # Desired Output: True or False
# Example:
    # Input: "The tiny turtle turned today"
    # Output: True

    # Input: "The rabbit and fox run fast"
    # Output: False

    # Input: "A big brown bear and a bold bird"
    # Output: True

def process(sentence: str) -> bool: 
    stop_words = ['a' , 'an', 'the', 'and']
    words = sentence.lower().split()

    # Cách loại bỏ stop word trong câu 
    significant_words = [word for word in words if word not in stop_words]

    # Nếu câu chỉ có 1 từ trả về True 
    if len(significant_words) <= 1: 
        return True
    
    # Lấy các chữ cái đầu tiên
    first_letter = significant_words[0][0]

    for word in significant_words: 
        if word[0] != first_letter:
            return False
        
    return True 

if __name__ == '__main__':
    text_01 = "The tiny turtle turned today"
    print(process(text_01))

    text_02 = "The rabbit and fox run fast"
    print(process(text_02))

    text_03 = "A big brown bear and a bold bird" 
    print(process(text_03))