import json


pers1 = {
    'age': 31,
    'sex': 'male',
    'city_country': 'Zurich, Switzerland',
    'birth_city_country': 'Cleveland, Ohio',
    'education': 'Masters in Computer Science',
    'occupation': 'software engineer',
    'income': '250 thousand swiss francs',
    'income_level': 'very high',
    'relationship_status': 'single'
}

pers2 = {
    'age': 45,
    'sex': 'female',
    'city_country': 'San Antonio, United States',
    'birth_city_country': 'San Antonio, United States',
    'education': 'High School Diploma',
    'occupation': 'shop owner',
    'income': '60 thousand us dollars',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers3 = {
    'age': 19,
    'sex': 'male',
    'city_country': 'London, United Kingdom',
    'birth_city_country': 'Budapest, Hungary',
    'education': 'studying towards a Bachelors in Economics',
    'occupation': 'part-time waiter',
    'income': ' 10 thousand pounds',
    'income_level': 'low',
    'relationship_status': 'in a relationship'
}

pers4 = {
    'age': 55,
    'sex': 'male',
    'city_country': 'Montreal, Canada',
    'birth_city_country': 'Montreal, Canada',
    'education': 'Bachelors in Business Administration',
    'occupation': 'financial manager',
    'income': '90 thousand canadian dollars',
    'income_level': 'middle',
    'relationship_status': 'divorced'
}

pers5 = {
    'age': 35,
    'sex': 'female',
    'city_country': 'Munich, Germany',
    'birth_city_country': 'Berlin, Germany',
    'education': 'Doctorate in Physics',
    'occupation': 'research scientist',
    'income': '100 thousand euros',
    'income_level': 'high',
    'relationship_status': 'single'
}

pers6 = {
    'age': 27,
    'sex': 'male',
    'city_country': 'Bogota, Colombia',
    'birth_city_country': 'Cali, Colombia',
    'education': 'Bachelors in Graphic Design',
    'occupation': 'graphic designer',
    'income': '20 million colombian pesos',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers7 = {
    'age': 48,
    'sex': 'female',
    'city_country': 'Cape Town, South Africa',
    'birth_city_country': 'Johannesburg, South Africa',
    'education': 'Masters in Public Health',
    'occupation': 'health inspector',
    'income': '450 thousand south african rand',
    'income_level': 'middle',
    'relationship_status': 'widowed'
}

pers8 = {
    'age': 22,
    'sex': 'female',
    'city_country': 'Helsinki, Finland',
    'birth_city_country': 'Turku, Finland',
    'education': 'studying towards a Bachelors in International Relations',
    'occupation': 'part-time tutor',
    'income': '15 thousand euros',
    'income_level': 'low',
    'relationship_status': 'single'
}

pers9 = {
    'age': 61,
    'sex': 'male',
    'city_country': 'Auckland, New Zealand',
    'birth_city_country': 'Christchurch, New Zealand',
    'education': 'High School Diploma',
    'occupation': 'retiree',
    'income': '20 thousand new zealand dollars',
    'income_level': 'low',
    'relationship_status': 'married'
}

pers10 = {
    'age': 33,
    'sex': 'female',
    'city_country': 'Beijing, China',
    'birth_city_country': 'Shanghai, China',
    'education': 'Masters in Architecture',
    'occupation': 'architect',
    'income': '260 thousand chinese yuan',
    'income_level': 'high',
    'relationship_status': 'single'
}

pers11 = {
    'age': 28,
    'sex': 'male',
    'city_country': 'Istanbul, Turkey',
    'birth_city_country': 'Ankara, Turkey',
    'education': 'Bachelors in Marketing',
    'occupation': 'marketing manager',
    'income': '60 thousand turkish lira',
    'income_level': 'middle',
    'relationship_status': 'engaged'
}

pers12 = {
    'age': 42,
    'sex': 'female',
    'city_country': 'Rome, Italy',
    'birth_city_country': 'Naples, Italy',
    'education': 'Doctorate in History',
    'occupation': 'university professor',
    'income': '75 thousand euros',
    'income_level': 'high',
    'relationship_status': 'married'
}

pers13 = {
    'age': 25,
    'sex': 'male',
    'city_country': 'Osaka, Japan',
    'birth_city_country': 'Tokyo, Japan',
    'education': 'Bachelors in Software Engineering',
    'occupation': 'junior software developer',
    'income': '6 million japanese yen',
    'income_level': 'middle',
    'relationship_status': 'single'
}

pers14 = {
    'age': 50,
    'sex': 'female',
    'city_country': 'Sydney, Australia',
    'birth_city_country': 'Perth, Australia',
    'education': 'Masters in Finance',
    'occupation': 'financial analyst',
    'income': '100 thousand australian dollars',
    'income_level': 'high',
    'relationship_status': 'divorced'
}

pers15 = {
    'age': 37,
    'sex': 'male',
    'city_country': 'Rio de Janeiro, Brazil',
    'birth_city_country': 'Sao Paulo, Brazil',
    'education': 'High School Diploma',
    'occupation': 'chef',
    'income': '45 thousand brazilian reais',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers16 = {
    'age': 29,
    'sex': 'female',
    'city_country': 'Stockholm, Sweden',
    'birth_city_country': 'Malmo, Sweden',
    'education': 'Bachelors in Computer Science',
    'occupation': 'web developer',
    'income': '55 thousand swedish krona',
    'income_level': 'middle',
    'relationship_status': 'in a relationship'
}

pers17 = {
    'age': 45,
    'sex': 'male',
    'city_country': 'Paris, France',
    'birth_city_country': 'Lyon, France',
    'education': 'Masters in Art History',
    'occupation': 'art curator',
    'income': '70 thousand euros',
    'income_level': 'high',
    'relationship_status': 'single'
}

pers18 = {
    'age': 39,
    'sex': 'female',
    'city_country': 'Edinburgh, United Kingdom',
    'birth_city_country': 'Glasgow, United Kingdom',
    'education': 'Doctorate in Astronomy',
    'occupation': 'astronomer',
    'income': '80 thousand pounds',
    'income_level': 'high',
    'relationship_status': 'married'
}

pers19 = {
    'age': 23,
    'sex': 'male',
    'city_country': 'New Delhi, India',
    'birth_city_country': 'Kolkata, India',
    'education': 'studying towards a Bachelors in Commerce',
    'occupation': 'part-time retail worker',
    'income': '200 thousand indian rupees',
    'income_level': 'low',
    'relationship_status': 'single'
}

pers20 = {
    'age': 56,
    'sex': 'female',
    'city_country': 'Madrid, Spain',
    'birth_city_country': 'Barcelona, Spain',
    'education': 'High School Diploma',
    'occupation': 'nurse',
    'income': '30 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'widowed'
}

pers21 = {
    'age': 29,
    'sex': 'male',
    'city_country': 'Dublin, Ireland',
    'birth_city_country': 'Cork, Ireland',
    'education': 'Masters in Data Science',
    'occupation': 'data scientist',
    'income': '70 thousand euros',
    'income_level': 'high',
    'relationship_status': 'single'
}

pers22 = {
    'age': 26,
    'sex': 'female',
    'city_country': 'Lisbon, Portugal',
    'birth_city_country': 'Porto, Portugal',
    'education': 'Bachelors in Visual Arts',
    'occupation': 'graphic designer',
    'income': '30 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'in a relationship'
}

pers23 = {
    'age': 52,
    'sex': 'male',
    'city_country': 'Helsinki, Finland',
    'birth_city_country': 'Turku, Finland',
    'education': 'Doctorate in Medicine',
    'occupation': 'surgeon',
    'income': '150 thousand euros',
    'income_level': 'high',
    'relationship_status': 'married'
}

pers24 = {
    'age': 68,
    'sex': 'male',
    'city_country': 'Sydney, Australia',
    'birth_city_country': 'Melbourne, Australia',
    'education': 'Bachelors in Business Administration',
    'occupation': 'retired CEO',
    'income': '80 thousand Australian dollars',
    'income_level': 'high',
    'relationship_status': 'widowed'
}

pers25 = {
    'age': 24,
    'sex': 'female',
    'city_country': 'Paris, France',
    'birth_city_country': 'Marseille, France',
    'education': 'Bachelors in Fashion Design',
    'occupation': 'fashion designer',
    'income': '40 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'single'
}

pers26 = {
    'age': 37,
    'sex': 'female',
    'city_country': 'Lusaka, Zambia',
    'birth_city_country': 'Ndola, Zambia',
    'education': 'Diploma in Nursing',
    'occupation': 'nurse',
    'income': '18 thousand Zambian kwacha',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers27 = {
    'age': 50,
    'sex': 'male',
    'city_country': 'New Delhi, India',
    'birth_city_country': 'Kolkata, India',
    'education': 'PhD in Astrophysics',
    'occupation': 'university professor',
    'income': '12 lakh Indian rupees',
    'income_level': 'middle',
    'relationship_status': 'divorced'
}

pers28 = {
    'age': 33,
    'sex': 'male',
    'city_country': 'Tokyo, Japan',
    'birth_city_country': 'Nagoya, Japan',
    'education': 'Masters in Computer Engineering',
    'occupation': 'game developer',
    'income': '7 million Japanese yen',
    'income_level': 'high',
    'relationship_status': 'single'
}

pers29 = {
    'age': 27,
    'sex': 'female',
    'city_country': 'Wellington, New Zealand',
    'birth_city_country': 'Auckland, New Zealand',
    'education': 'Bachelors in Environmental Sciences',
    'occupation': 'environmental consultant',
    'income': '55 thousand New Zealand dollars',
    'income_level': 'middle',
    'relationship_status': 'in a relationship'
}

pers30 = {
    'age': 62,
    'sex': 'male',
    'city_country': 'Oslo, Norway',
    'birth_city_country': 'Bergen, Norway',
    'education': 'Masters in Structural Engineering',
    'occupation': 'structural engineer',
    'income': '600 thousand Norwegian Krone',
    'income_level': 'high',
    'relationship_status': 'married'
}

pers31 = {
    'age': 41,
    'sex': 'female',
    'city_country': 'Rome, Italy',
    'birth_city_country': 'Bologna, Italy',
    'education': 'Diploma in Gastronomy',
    'occupation': 'chef',
    'income': '35 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'divorced'
}

pers32 = {
    'age': 58,
    'sex': 'female',
    'city_country': 'Cape Town, South Africa',
    'birth_city_country': 'Durban, South Africa',
    'education': 'Masters in Education',
    'occupation': 'high school principal',
    'income': '400 thousand South African rand',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers33 = {
    'age': 23,
    'sex': 'male',
    'city_country': 'New York, United States',
    'birth_city_country': 'Los Angeles, United States',
    'education': 'Currently studying Bachelors in Film Studies',
    'occupation': 'part-time film editor',
    'income': '20 thousand US dollars',
    'income_level': 'low',
    'relationship_status': 'single'
}

pers34 = {
    'age': 47,
    'sex': 'female',
    'city_country': 'Toronto, Canada',
    'birth_city_country': 'Montreal, Canada',
    'education': 'Masters in Psychology',
    'occupation': 'psychologist',
    'income': '85 thousand Canadian dollars',
    'income_level': 'middle',
    'relationship_status': 'divorced'
}

pers35 = {
    'age': 36,
    'sex': 'male',
    'city_country': 'Madrid, Spain',
    'birth_city_country': 'Barcelona, Spain',
    'education': 'Bachelors in History',
    'occupation': 'museum curator',
    'income': '40 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers36 = {
    'age': 55,
    'sex': 'female',
    'city_country': 'Berlin, Germany',
    'birth_city_country': 'Hamburg, Germany',
    'education': 'PhD in German Literature',
    'occupation': 'college professor',
    'income': '70 thousand euros',
    'income_level': 'middle',
    'relationship_status': 'widowed'
}

pers37 = {
    'age': 30,
    'sex': 'female',
    'city_country': 'Gothenburg, Sweden',
    'birth_city_country': 'Stockholm, Sweden',
    'education': 'Masters in Architecture',
    'occupation': 'architect',
    'income': '550 thousand Swedish Krona',
    'income_level': 'middle',
    'relationship_status': 'in a relationship'
}

pers38 = {
    'age': 21,
    'sex': 'male',
    'city_country': 'Amsterdam, Netherlands',
    'birth_city_country': 'Rotterdam, Netherlands',
    'education': 'Studying Bachelors in Graphic Design',
    'occupation': 'part-time graphic designer',
    'income': '15 thousand euros',
    'income_level': 'low',
    'relationship_status': 'single'
}

pers39 = {
    'age': 39,
    'sex': 'female',
    'city_country': 'Mexico City, Mexico',
    'birth_city_country': 'Guadalajara, Mexico',
    'education': 'Masters in Business Administration',
    'occupation': 'business development manager',
    'income': '400 thousand Mexican pesos',
    'income_level': 'middle',
    'relationship_status': 'married'
}

pers40 = {
    'age': 51,
    'sex': 'male',
    'city_country': 'Buenos Aires, Argentina',
    'birth_city_country': 'Mendoza, Argentina',
    'education': 'Law Degree',
    'occupation': 'lawyer',
    'income': '800 thousand Argentine Peso',
    'income_level': 'middle',
    'relationship_status': 'divorced'
}

profiles = {f'pers{i+1}': eval(f'pers{i+1}') for i in range(40)}

with open('user_bot_profiles_40.json', 'w') as f:
    json.dump(profiles, f)
