# Question 2: (2 marks)
# Write a Python program that inputs a text file and outputs a new file where all the words are converted to lowercase.
# Input:

# A text file named "upper.txt" with the following content:
# THIS IS A SAMPLE TEXT FILE.
# ALL LETTERS ARE UPPERCASE.

# Desired Output:
# A new file named "lower.txt" with the following content:
# this is a sample text file.
# all letters are lowercase.


def convert_to_lowercase(input_file: str, output_file: str) -> None:
    try:
        with open(input_file, "r") as infile:
            content = infile.read()

        lowercase_content = content.lower()

        with open(output_file, "w") as outfile:
            outfile.write(lowercase_content)

        print(f"Successfully converted {input_file} to lowercase and saved as {output_file}")

    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")

def main() -> None:
    input_filename = "upper.txt"
    output_filename = "lower.txt"

    convert_to_lowercase(input_filename, output_filename)

if __name__ == "__main__":
    main()