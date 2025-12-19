# Question 20: (3 marks)
# Write a program to find the longest common prefix among all words in a text.
# If there's no common prefix, return an empty string.

# Input 1: "flower flowing flight"
# Output 1: "fl"

# Input 2: "dog cat bird"
# Output 2: ""

# Input 3: "interspecies interstellar interstate"
# Output 3: "inters"

def longest_common_prefix(text: str) -> str:
    # TODO: Implement your solution here
    pass

if __name__ == '__main__':
    test1 = "flower flowing flight"
    test2 = "dog cat bird"
    test3 = "interspecies interstellar interstate"
    
    print(f"'{test1}': '{longest_common_prefix(test1)}'")
    print(f"'{test2}': '{longest_common_prefix(test2)}'")
    print(f"'{test3}': '{longest_common_prefix(test3)}'")
