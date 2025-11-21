flowers = ['camellia', 'pendulum', 'petunias', 'begonia', 'dahlia', 'hostas', 'pelorism', 'paperwhite']

# a. Print all words ending with lia
words_ending_with_lia = [word for word in flowers if word.endswith('lia')]
print(f"a. {words_ending_with_lia}")

# b. Print all words longer than eight characters
words_longer_than_eight = [word for word in flowers if len(word) > 8]
print(f"b. {words_longer_than_eight}")

