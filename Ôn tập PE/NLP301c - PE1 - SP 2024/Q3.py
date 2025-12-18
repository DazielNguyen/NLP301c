# Q3: Write a program to extract the FPT University email addresses present in the text.
# Input:
# text = Please contact us at contact@fpt.edu.vn for further information. You can also
# give feedback at feedback@gmail.com'
# Output:
# ['contact@fpt.edu.vn']

def process(sentence: str) -> list:
    words = sentence.split(' ')
    output = [word for word in words if word.__contains__('@fpt')]
    return output

if __name__ == '__main__': 
    text = "Please contact us at contact@fpt.edu.vn for further information. You can also give feedback at feedback@gmail.com"
    print(process(text))