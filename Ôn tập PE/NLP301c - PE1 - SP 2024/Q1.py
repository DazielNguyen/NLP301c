# Đề PE1_SP24_89588
# #Q1: Define flowers to be the list of words 
# ['camellia', 'pendulum", 'petunias", 'begonia", 'dahlia', 'hostas", 'pelorism", paperwhite'].
# # Now write code to perform the following tasks:
# # a. Print all words ending with lia
# # b. Print all words longer than eight characters
#
# # Input: "'camellia', 'pendulum", 'petunias", begonia", 'dahlia", 'hostas", 'pelorism", 'paperwhite'"
#
# # Desired Output:
# # a. ['camellia', 'dahlia']
# # b. ['paperwhite']


def process(flowers: list) -> list: 
    a = [word for word in flowers if word.endswith('lia')]
    print(f'a. {a}')

    b = [word for word in flowers if len(word) > 8]
    print(f'b. {b}')

if __name__ == '__main__': 
    flowers = ['camellia', 'pendulum', 'petunias', 'begonia', 'dahlia', 'hostas', 'pelorism', 'paperwhite']
    process(flowers)

