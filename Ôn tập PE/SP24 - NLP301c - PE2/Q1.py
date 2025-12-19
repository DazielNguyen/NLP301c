# Question 1: (2 marks)
# Write code to convert nationality adjectives like Canadian and Australian to the 
# corresponding nouns Canada and Australia
# Example:
# Input:
# ['Argentinian', 'Australian', ' Canadian']
# Output:
# 'Argentina, 'Australia', Canada"]

def convert_to_country(nationalities):
    countries = {
        "Argentinian": "Argentina",
        "Australian": "Australia",
        "Canadian": "Canada",
        "American": "United States",
        "British": "United Kingdom",
        "Chinese": "China",
        "French": "France",
        "German": "Germany",
        "Indian": "India",
        "Italian": "Italy",
        "Japanese": "Japan",
        "Russian": "Russia",
        "Spanish": "Spain",
        "Vietnamese": "Vietnam",
        "Brazilian": "Brazil",
        "Mexican": "Mexico",
    }

    return [countries.get(n, n) for n in nationalities]

input = ['Argentinian', 'Australian', 'Canadian']

print(convert_to_country(input))