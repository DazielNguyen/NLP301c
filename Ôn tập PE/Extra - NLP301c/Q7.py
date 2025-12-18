# Question 7: (3 marks)
# Write a program that converts snake_case strings to camelCase.
# Input: A string in snake_case format.
# Desired Output: A string in camelCase format.
# 
# Example:
# Input: "hello_world_from_python"
# Output: "helloWorldFromPython"
# 
# Input: "natural_language_processing"
# Output: "naturalLanguageProcessing"
# 
# Hints:
# - Split by underscore '_'
# - First word stays lowercase
# - Capitalize first letter of subsequent words
# - Use str.capitalize() or str.title()

def snake_to_camel(snake_str) -> str:
    """
    TODO: Implement this function
    1. Split the string by underscore
    2. Keep first word as is (lowercase)
    3. Capitalize first letter of remaining words
    4. Join all words together
    """
    words = snake_str.split('_')
    first_word = words[0].lower()
    capitalize_word = [word.capitalize() for word in words[1:]]
    result = ' '.join([first_word]+capitalize_word)
    return result


# Test cases
if __name__ == "__main__":
    # Test 1
    text1 = "Hello_world_from_python"
    print(snake_to_camel(text1))  # Expected: "helloWorldFromPython"
    
    # Test 2
    text2 = "natural_language_processing"
    print(snake_to_camel(text2))  # Expected: "naturalLanguageProcessing"
    
    # Test 3
    text3 = "simple_test"
    print(snake_to_camel(text3))  # Expected: "simpleTest"
    
    # Test 4
    text4 = "single"
    print(snake_to_camel(text4))  # Expected: "single"
