ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or "
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "SCHL_RED": {
        0.0: "No High School",
        1.0: "High School Diploma",
        2.0: "Associate's Degree",
        3.0: "Bachelor's Degree",
        4.0: "Master's Degree",
        5.0: "Professional Degree",
        6.0: "Doctorate Degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White",
        2.0: "Black or African American",
        3.0: "American Indian",
        4.0: "Alaska Native",
        5.0: "American Indian or Alaska Native tribes specified",
        6.0: "Asian",
        7.0: "Native Hawaiian and Other Pacific Islander",
        8.0: "Some Other Race",
        9.0: "Two or More Races",
    },
    "RELP": {
        0.0: "Reference person",
        1.0: "Husband/wife",
        2.0: "Biological son or daughter",
        3.0: "Adopted son or daughter",
        4.0: "Stepson or stepdaughter",
        5.0: "Brother or sister",
        6.0: "Father or mother",
        7.0: "Grandchild",
        8.0: "Parent-in-law",
        9.0: "Son-in-law or daughter-in-law",
        10.0: "Other relative",
        11.0: "Roomer or boarder",
        12.0: "Housemate or roommate",
        13.0: "Unmarried partner",
        14.0: "Foster child",
        15.0: "Other nonrelative",
        16.0: "Institutionalized group quarters population",
        17.0: "Noninstitutionalized group quarters population",
    },
    "CIT": {
        1.0: "Born in the U.S.",
        2.0: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
        3.0: "Born abroad of American parent(s)",
        4.0: "U.S. citizen by naturalization",
        5.0: "Not a citizen of the U.S.",
    },
    "JWTR": {
        0.0: "N/A",
        1.0: "Car, truck, or van",
        2.0: "Bus or trolley bus",
        3.0: "Streetcar or trolley car (carro publico in Puerto Rico)",
        4.0: "Subway or elevated",
        5.0: "Railroad",
        6.0: "Ferryboat",
        7.0: "Taxicab",
        8.0: "Motorcycle",
        9.0: "Bicycle",
        10.0: "Walked",
        11.0: "Worked at home",
        12.0: "Other method",
    },
    "PINCP": {
        0.0: "Under $50000",
        1.0: "Over $50000",
    },
}


def get_acs_cat(feature: str, value: float, reduced: bool = False) -> str:
    if reduced:
        return ACSIncome_categories[f"{feature}_RED"][value]
    return ACSIncome_categories[feature][value]


def get_inv_acs_cat(feature: str, value: str) -> float:
    return {v: k for k, v in ACSIncome_categories[feature].items()}[value]
