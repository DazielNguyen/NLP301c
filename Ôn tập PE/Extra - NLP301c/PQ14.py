# Question 14: (3 marks)
# Write a program to convert snake_case strings to camelCase.
# If input is already in camelCase or other format, handle it appropriately.

# Input:
# ["hello_world", "natural_language_processing", "python_is_fun", "already_in_camelCase"]

# Desired Output:
# ['helloWorld', 'naturalLanguageProcessing', 'pythonIsFun', 'alreadyInCamelcase']

def snake_to_camel(names: list) -> list:
    # TODO: Implement your solution here
    result = []
    
    for name in names:
        # Tách theo underscore
        parts = name.split('_')
        
        # Nếu không có underscore, giữ nguyên nhưng lowercase
        if len(parts) == 1:
            result.append(name.lower())
        else:
            # Phần đầu tiên giữ lowercase, các phần khác capitalize
            camel = parts[0].lower()
            for part in parts[1:]:
                if part:  # Kiểm tra không rỗng
                    camel += part.capitalize()
            result.append(camel)
    
    return result

if __name__ == '__main__':
    names = ["hello_world", "natural_language_processing", "python_is_fun", "already_in_camelCase"]
    print(snake_to_camel(names))
