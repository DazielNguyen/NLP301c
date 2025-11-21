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

    return [countries.get(nationality, nationality) for nationality in nationalities]


input_nationalities = ["Argentinian", "Australian", "Canadian"]
output_countries = convert_to_country(input_nationalities)
print(output_countries)
