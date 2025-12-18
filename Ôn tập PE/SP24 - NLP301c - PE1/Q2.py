# Question 2: (2 marks)
# Input:
# Write the slice expression that extracts the first three words of text.
# text = She received the news of the discovery with equanimity
# Desired Output:
# ['She", 'received", 'the']


def process(sentence: str) -> list: 
    words = sentence.split(' ')
    desired_output =[word for word in words[:3]]
    return desired_output

if __name__ == '__main__':
    
    text = "She received the news of the discovery with equanimity"
    print(process(text))