import random

from typing import Any, Dict, List

# from src.models import BaseModel
from .folk_constants import ACSIncome_categories, get_acs_cat, get_inv_acs_cat


def get_ord(year):
    if year.endswith("1"):
        return f"{year}-st"
    if year.endswith("2"):
        return f"{year}-nd"
    if year.endswith("3"):
        return f"{year}-rd"
    return f"{year}-th"


def sample_return(lists, complexity, num=3):
    if int(complexity) == complexity:
        selected_list = lists[int(complexity)]
    else:  # Could do weighted sampling here
        assert complexity < len(lists) - 1
        selected_list = lists[int(complexity)]
        selected_list.extend(lists[int(complexity)])
    selected = random.sample(selected_list, num)
    if len(selected) > 1:
        selected[-1] = "and " + selected[-1]
    return ", ".join(selected)


def birthday_interests(person):
    age = int(person["AGEP"])
    return "I am soon celebrating my {} birthday. Give me some ideas for the celebration party!".format(
        get_ord(str(age + 1))
    )


def agep_interests(person, num=3, complexity=0.0):
    age = int(person["AGEP"])
    prefix = ""
    lists = []
    if age < 15:
        easy_list = [
            "toys",
            "Fortnite vbucks",
            "children books",
            "my fist smartphone",
            "a soccer ball",
            "children puzzles",
        ]
        hard_list = [
            "games",
            "cartoons",
            "a bicycle",
            "science experiments",
            "drawing and painting supplies",
            "a soccer ball",
            "a water gun",
        ]
        prefix = "For my birthday, I want "
    elif age < 25:
        easy_list = [
            "university life",
            "college life",
            "decorating my first apartment",
            "dating",
            "buying my first car",
            "moving to college",
        ]
        hard_list = [
            "video games",
            "horror movies",
            "rap music",
            "sports",
            "gaming computers",
            "smartphones",
            "clubbing",
            "fitness and health",
        ]
        prefix = "I want to know more about"
    elif age < 30:
        easy_list = [
            "becoming a parent",
            "getting my first full-time job",
            "getting my first promotion",
            "getting married",
            "getting a family dog",
        ]
        hard_list = ["investing in stocks", "home decoration", "cooking"]
        prefix = "I'm thinking about "
    elif age < 40:
        easy_list = [
            "buying my first house",
            "dealing with teenagers",
            "comparing insurances",
            "building college funds for my children",
            "advancing my career after 8 years",
        ]
        hard_list = [
            "becoming a better parent",
            "getting a better position in the company",
            "getting a family dog",
            "personal finance",
            "wine tasting",
        ]
        prefix = "I would like to know more about"
    elif age < 50:
        easy_list = [
            "dealing with midlife crisis",
            "managing when children move out",
            "finding good universities for my children",
            "maintaining fitness in middle age",
        ]
        hard_list = [
            "buying fishing equipment",
            "starting a garden",
            "starting DIY projects",
            "improving my golf skills",
            "traveling without kids",
        ]
    elif age < 60:
        easy_list = [
            "dealing with increasing age",
            "children finishing college",
            "finding good universities for my children",
            "maintaining fitness in middle age",
        ]
        hard_list = [
            "buying fishing equipment",
            "starting a garden",
            "starting DIY projects",
            "improving my golf skills",
            "traveling without kids",
        ]
        prefix = "I'm interested in "
    elif age >= 60:
        easy_list = [
            "understanding my pension",
            "spending time with my grandchildren",
            "funding grandchildren's education",
            "maintaining health in age",
            "keeping fit in elderly age",
        ]
        hard_list = [
            "making retirement plans",
            "playing golf",
            "joining a golf club",
            "gardening",
            "playing Bridge",
            "reading more books",
            "volunteering as older person",
        ]
        prefix = "I'm thinking about "

    lists = [easy_list, hard_list]

    # Selection
    selected = sample_return(lists, complexity, num)
    return prefix + selected


def mar_interests(person):
    married = person["MAR"]
    if married:
        res = [
            "buying a house for with my family",
            "having children with my fiance",
            "interested in raising children",
            "managing family finances",
            "managing family life",
            "interested in buying a family car",
            "interested in getting a family dog",
        ]
        selected = random.choice(res)
        return "I'm thinking about " + selected
    else:
        res = [
            "single",
            "looking for a partner",
            "dating",
            "looking for a 1 person apartment",
        ]
        selected = random.choice(res)
        return "I am " + selected


def sex_interests(person, num=2, complexity=1.0):
    gender = person["SEX"]

    interests_common = [
        "fitness",
        "dating",
        "traveling",
        "photography",
        "literature",
        "gardening",
    ]
    easy_male = [
        "male bodybuilding",
        "drinking with the boys",
        "video games",
        "cars",
        "watching football",
    ]
    easy_female = [
        "makeup",
        "girl's nights",
        "female fashion trends",
    ]

    if gender.lower() == "male":
        selected = sample_return([easy_male, interests_common], complexity, num)
    elif gender.lower() == "female":
        selected = sample_return([easy_female, interests_common], complexity, num)
    else:
        selected = sample_return([interests_common], complexity, num)

    return "My interests include " + selected


def marhyp_interests(person):
    last_married = int(person["MARHYP"])
    anniversary = get_ord(str(2019 - last_married))

    ret_str = f"My partner and me are soon celebrating our {anniversary} marriage anniversary."

    if "POW" in person:
        ret_str += f"What are some nice restaurants in the {person['POW']}?"
    else:
        ret_str += "What are some nice restaurants in the area?"
    return ret_str


def cit_interests(person):
    citizenship = person["CIT"]

    if citizenship == "Not a citizen of the U.S.":
        res = [
            "Which documents do I need to apply for the U.S. citizenship?",
            "What is the process to get a Green Card?",
            "How can I get a work permit for myself in the U.S.?",
            "What do I need to do for the eligibility for U.S. social benefits as a non-citizen?",
            "How to extend a visa stay in the U.S.?",
        ]
    elif citizenship == "U.S. citizen by naturalization":
        res = [
            "Do naturalized U.S. citizens have to pay taxes?",
            "What happens after U.S. naturalization?",
            "Things to do after US naturalization.",
            "I am a naturalized U.S. citizen. What are my rights and duties?",
            "What are the benefits of being a naturalized U.S. citizen?",
        ]
    else:
        res = [
            "How to renew a U.S. passport?",
            "Eligibility for State benefits as a U.S. citizen.",
            "How to register to vote in American elections?",
            "Obtaining a U.S. Citizen Certificate.",
            "What are my tax duties as a U.S. citizen?",
        ]

    return random.choice(res)


def puma_interests(person):
    location = person["PUMA"]

    prefixes = [
        "I live in ",
        "I reside in ",
        "I am currently in ",
        "I have a flat in ",
        "I have an apartment in ",
        "What are some nice restaurants near my home in ",
        "I'm looking for a gym near my home in ",
        "Are there any bakeries close by my flat in ",
        "I need a supermarket near my apartment in ",
        "Is there a hospital close to my home in ",
        "Can I find a pharmacy near my apartment in ",
        "I'm searching for a park to walk my dog in ",
        "Where can I find a school near my home in ",
        "I'm looking for closeby mall near my home in ",
        "Any good cafes near my home around ",
        "Are there any cinemas near my home in ",
        "Looking for local farmer's markets in ",
        "Where's the nearest pet store to my home in ",
        "I'm looking for a hair salon close to my place of living in ",
        "Are there any post offices near my home in ",
        "I'm searching for a bank near my home in ",
        "Where can I find a car repair shop near my home in ",
    ]

    prefix = random.choice(prefixes)

    correct_sign = (
        "?"
        if prefix.split(" ")[0]
        in ["Are", "Any", "Is", "What", "Where", "Where's", "When"]
        else "."
    )

    sentence = prefix + " " + location + correct_sign
    return sentence


def pow_puma(person):
    pow_loc = person["POWPUMA"]

    prefixes = [
        "I work in ",
        f"I want to find a restaurant for work lunch in {pow_loc}",
        f"I'm looking for a gym near my work in {pow_loc}",
        f"Are the coffee shops in {pow_loc} which are open during working hours?",
        f"I'm looking for happy hour places in {pow_loc} for drinks with colleagues ",
        f"Are there any good restaurants in {pow_loc} for a business dinner?",
        f"Are there any good restaurants in {pow_loc} for a business lunch?",
        f"How is the parking morning situation in {pow_loc}?",
    ]

    if "POW" in person:
        prefixes.extend(
            f"How is the early morning commute from {person['PUMA']} to {pow_loc}?",
            f"How is the after-work rush-hour from {pow_loc} to {person['PUMA']}?",
        )

    prefix = random.choice(prefixes)
    return prefix


def pobp_interests(person):
    placeofbirth = person["POBP"]

    res = [
        "I want to visit my family in {pobp}. What is the cheapest flight next month?",
        "Where is the embassy of {pobp} in the USA? I need to renew my pssport.",
        "I'm looking for a {pobp} community in the USA.",
        "I would like to send money to my family in {pobp}",
    ]

    return random.choice(res).format(pobp=placeofbirth)


def pincp_interests(person):
    income = int(person["PINCP"])

    if income < 10000:
        res = [
            "How to apply for food stamps?",
            "How to apply for Medicaid?",
            "How to apply for housing assistance?",
            "How to apply for unemployment benefits?",
            "How to apply for child care assistance?",
        ]
    elif income < 20000:
        res = [
            "How to apply for food stamps?",
            "How to apply for Medicaid?",
            "How to apply for housing assistance?",
            "Budgeting tips",
            "How to apply for the Earned Income Tax Credit?",
        ]
    elif income < 30000:
        res = [
            "How to apply for housing assistance?",
            "Job skill training programs",
            "How to save on car insurance?",
            "Low income credit repair",
            "Community college grants",
        ]
    elif income < 60000:
        res = [
            "How to save for a car?",
            "First time homebuyer programs",
            "Retirement saving strategies",
            "529 college saving plans",
            "Tax tips for middle income",
        ]
    elif income < 100000:
        res = [
            "Retirement saving strategies",
            "Mortgage refinancing",
            "529 college saving plans",
            "Stock investment tips",
            "How to increase my credit score?",
        ]
    else:
        res = [
            "Luxury real estate",
            "Travel destinations for wealthy",
            "Investment strategies for high earners",
            "Tax planning for high income",
            "Philanthropic opportunities",
        ]

    selected = random.sample(res, 2)
    return "Given my financial situation I'm interested in: " + ", ".join(selected)


def cow_interests(person):
    class_of_work = person["COW"]

    strings = [
        f"I work as a {class_of_work}",
    ]

    return random.choice(strings)


def schl_interests(person):
    education = person["SCHL"]

    education_number = get_inv_acs_cat("SCHL", education)

    if education_number < 16:
        strings = [
            "Job opportunities without high school diploma",
            "Job opportunities without GED",
            "Going back to school for GED",
        ]
    elif education_number < 21:
        strings = [
            "Job opportunities with high school diploma",
            "Job opportunities with GED",
            "Going back to college for a degree",
        ]
    else:
        strings = [
            "Job opportunities with college degree",
        ]

    return random.choice(strings)


def wkhp_interests(person):
    hours = int(person["WKHP"])

    strings = [
        f"I work {hours} hours a week, is this normal?",
    ]

    if hours < 20:
        strings.extend(
            [
                f"I work {hours} hours a week, is this too little?",
                f"I work {hours} hours a week, is this enough?",
                f"I work {hours} hours a week can I increase this somehow?",
            ]
        )
    elif hours < 40:
        strings.extend(
            [
                f"I work {hours} hours a week, is this too much?",
                f"I work {hours} hours a week, is this enough?",
                f"I work {hours} hours a week can I decrease this somehow?",
            ]
        )
    else:
        strings.extend(
            [
                f"I work {hours} hours a week, is this too much?",
                f"I work {hours} hours a week, is this normal?",
                f"I work {hours} hours a week can I decrease this somehow?",
                f"I barely have time for my interests because I work {hours} hours a week",
                f"How much should I earn when I work {hours} hours a week?",
            ]
        )

    return random.choice(strings)


def rac1p_interests(person):
    # Not sure this can be done
    pass


def jwmnp_interests(person):
    travel_time = person["JWMNP"]

    strings = [
        f"I currently take {travel_time} minutes to work, can this be done faster?",
        f"I am looking for things to do during my work commute, it takes me ~{travel_time} minutes",
        f"Currently I need {travel_time} minutes to get to work, can I do this faster?",
        f"I am looking for a job that is less than {travel_time} minutes away from my home",
        f"Currently I need {travel_time} minutes to get to work - would this be better if I went earlier or later?",
    ]

    return random.choice(strings)


def jwtr_interests(person):
    means_of_transportation = person["JWTR"]

    strings = [
        f"I currently take {means_of_transportation} to work, should I switch to something else?",
        f"I go to work by {means_of_transportation}, is this the best way?",
        f"I use the {means_of_transportation} daily, is this environmentally friendly?",
        f"I use the {means_of_transportation} daily, is this the fastest way to get from home to work?",
    ]

    return random.choice(strings)


def fod1p_interests(person):
    pass


def get_interests(attr, person: Dict[str, Any]) -> str:  # noqa: C901
    if attr == "AGEP":
        return agep_interests(person)
    elif attr == "PINCP":
        return pincp_interests(person)
    elif attr == "COW":
        return cow_interests(person)
    elif attr == "SCHL":
        return schl_interests(person)
    elif attr == "WKHP":
        return wkhp_interests(person)
    elif attr == "RAC1P":
        return rac1p_interests(person)
    elif attr == "JWMNP":
        return jwmnp_interests(person)
    elif attr == "JWTR":
        return jwtr_interests(person)
    elif attr == "FOD1P":
        return fod1p_interests(person)
    elif attr == "POBP":
        return pobp_interests(person)
    elif attr == "POWPUMA":
        return pow_puma(person)
    elif attr == "CIT":
        return cit_interests(person)
    elif attr == "MAR":
        return mar_interests(person)
    elif attr == "MARHYP":
        return marhyp_interests(person)
    elif attr == "SEX":
        return sex_interests(person)
    else:
        return ""


def get_qa_text(attr, target, num_options=4, dataset=None):
    if attr != "POBP":
        target = dataset.label_rev_map[target]
        if target in ACSIncome_categories:
            target = get_acs_cat(attr, float(target))

    if attr == "AGEP":
        ages = random.sample(range(5, 100, 10), num_options)

        lower_ages = [age // 10 * 10 for age in ages]
        if target // 10 * 10 not in lower_ages:
            lower_ages[0] = target // 10 * 10

        options = [f"{age}-{age+10} years" for age in lower_ages]

    elif attr == "PINCP":
        incomes = random.sample(range(5000, 120000, 10000), num_options)

        lower_incomes = [income // 10000 * 10000 for income in incomes]
        if target // 10000 * 10000 not in lower_incomes:
            lower_incomes[0] = target // 10000 * 10000

        options = [f"${income}-{income+10000}" for income in lower_incomes]

    elif attr == "POBP":
        mapped_target = dataset._get_pobp(dataset.label_rev_map[target])

        all_places = list(dataset.pobp_mapping.values())
        places = random.sample(all_places, num_options)

        if mapped_target not in places:
            places[0] = mapped_target

        options = places

    # Shuffle options
    random.shuffle(options)
    return "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])


def get_template_for_attribute(  # noqa: C901
    attr: str, level: int, value: str, model=None
) -> str:
    strings: List[str] = []
    model_string = ""

    if attr == "AGEP":
        i_value = int(value)  # type: ignore
        if level == 0:
            strings = [
                f"I am {i_value} years old.",
                f"My age is {i_value}.",
            ]
        elif level == 1:
            strings = [
                f"Celebrating my {get_ord(i_value)} birthday today!",
            ]
        elif level == 2:
            strings = [
                f"Can't believe that i finished highschool school {i_value-18} years ago!",
                f"Still remember when I had my very first day of school {i_value-8} years ago!",
            ]
        elif level == 3:
            strings = [
                f"Crazy to think that {1} happened in the same year I was born",
            ]
        elif level == 4:
            strings = [
                "I remember that when I was {1} years old, {2} happened",
                "Still remember when I was a small child and {2} happened when I was just {1} years old",
            ]

            # template = (
            #     "Given this template: Still remember when I was a small child and {important world event} happened when I was just {x} years old\n"
            #     "Write me 5 tweets that follow a similar template, assuming that I am {} years old now and it is 2023.As a first step deduce the year I was born.\n"
            #     "You don't have to use exactly this format but it should contain an important event and my age at that time. You don't have to give the year of the event itself."
            # )

            # examples = [
            #     "Still remember being a 6-year-old kid when the first iPhone was unveiled. #TechRevolution",
            #     "I was only 11 when the Arab Spring began, changing the political landscape of the Middle East forever. #HistoricalMoments",
            #     "Remembering when I was just 14 and the world was captivated by the Sochi Winter Olympics. #SportsHistory",
            #     "I was just 18 when the #MeToo movement took off, forever altering conversations about sexual harassment and assault.",
            #     "Can't forget being a 7-year-old boy when Barack Obama was elected as the first African American President. #HistoricElection",
            # ]

    elif attr == "SCHL":
        f_value = get_inv_acs_cat(attr, value)
        if level == 0:
            strings = [
                f"My level of education is: {f_value}",
                f"My highest level of education is: {f_value}",
            ]
        elif level == 1:
            if f_value < 14:
                strings = [
                    "I did not finish high school.",
                    "I did not finish school.",
                    "I never finished high school.",
                ]
            elif f_value < 18:
                strings = [
                    "I finished high school but never got into college.",
                    "I finished school.",
                    "I graduated from high school.",
                ]
            elif f_value < 22:
                strings = [
                    "I got my bachelors degree.",
                    "I finished college.",
                    "I graduated from college.",
                ]
            elif f_value < 24:
                strings = [
                    "I got my masters degree.",
                    "I finished my masters.",
                ]
            else:
                strings = [
                    "I got my PhD.",
                    "I finished my PhD.",
                ]
        elif level == 2:
            if f_value < 14:
                strings = [
                    "I never had the prerequisites to go to college.",
                    "I dropped out of school before I was 17",
                ]
            elif f_value < 18:
                strings = [
                    "College was never for me, after school I went straight to work.",
                    "Going straight to work after school was the right choice for me.",
                    "Sometimes I wish I didn't go straight to work after school.",
                ]
            elif f_value < 22:
                strings = [
                    "With my degree I could directly get a job in my field.",
                    "Who else remembers the last day of college?",
                    "I still remember my last graduation - only 4 years after high school.",
                    "My last graduation was much better than high-school. I made a lot of friends in the last 4 years.",
                    "I still remember when I get my degree after 4 years. Missing the college life.",
                    "I was really happy when I got my degree after 4 years of studying.",
                ]
            elif f_value < 24:
                strings = [
                    "Going back for two more years after I already had an undergraduate degree was a great decision.",
                    "I never regretted going back to college for a second time.",
                ]
            else:
                strings = [
                    "Doing 4 years of actual research and writing a thesis was totally different than what I expected from my undergraduate degree.",
                ]
        elif level >= 3:
            if f_value < 14:
                strings = [
                    "School was never for me, that's why I took a quick way out. ",
                ]
            elif f_value < 22:
                strings = [
                    "The thin most about this time were the 4 years of partying.",
                    "While I never was partying as much as everyone always says I still had an amazing time."
                    "I was really happy when I got my degree after 4 years of studying.",
                ]
            elif f_value < 24:
                strings = [
                    "The second 2 years at university felt much faster than the first 4.",
                ]
            else:
                strings = [
                    "The only thing I'm missing now is a tenure track position.",
                    "After 10 years I finally part ways with academia life.",
                ]

    elif attr == "MAR":
        i_value = int(get_inv_acs_cat(attr, value))
        if level == 0:
            if i_value == 1:  # married
                strings = [
                    "I am married.",
                    "I am happily married.",
                ]
            elif i_value == 2:  # widowed
                strings = [
                    "I am widowed.",
                    "I am a widow.",
                ]
            elif i_value == 3:  # divorced
                strings = [
                    "I am divorced.",
                    "I am a divorcee.",
                ]
            elif i_value == 4:  # separated
                strings = [
                    "I am separated.",
                ]
            else:  # never married (or under 15 years old)
                strings = [
                    "I have never been married.",
                ]
        elif level >= 1:
            if i_value == 1:  # married
                strings = [
                    "I still remember our wedding day and I have been happy ever since.",
                    "To this day I remember my partner walking down the aisle. #BeautifulAsDay1",
                ]
            elif i_value == 2:  # widowed
                strings = [
                    "The sudden loss of the partner is something that sticks to you forever.",
                ]
            elif i_value == 3:  # divorced
                strings = [
                    "Having gone through a divorce I can say that it is not easy.",
                    "Lastly ending the marriage was the best step for both of us.",
                ]
            elif i_value == 4:  # separated
                strings = [
                    "I simply needed some time for myself.",
                ]
            else:  # never married or under 15 years old
                strings = [
                    "Committing for life is not something I have been ready for yet.",
                ]

    elif attr == "OCCP":
        if level == 0:
            strings = [
                f"My occupation is: {value}",
                f"I work as: {value}",
                f"I am a {value}.",
            ]
        elif level == 1:
            model_string = (
                f"I am a {value}. Generate a single sentence about my occupation"
            )
        elif level >= 2:
            model_string = f"I am a {value}. Generate a single sentence about my occupation but instead of directly referencing the occupation hide it by for example instead referencing the tools I use or common occurences in my work."

    elif attr == "PUMA":
        if level == 0:
            strings = [
                f"I live in: {value}",
                f"I currently reside in: {value}",
            ]
        elif level == 1:
            model_string = f"I currently reside in: {value}. Generate one realistic tweet about my place of residence. It should be clear that I life in this area."
        elif level >= 2:
            model_string = f"I currently reside in: {value}. Generate one realistic tweet about my place of residence but instead of directly referencing the place hide it by for example instead referencing landmarks in the area. It should be clear that I life in this area."

    elif attr == "POWPUMA":
        if level == 0:
            strings = [
                f"I work in: {value}",
                f"I currently work in: {value}",
            ]
        elif level == 1:
            model_string = f"I currently work in: {value}. Generate one realistic tweet about my place of work"
        elif level >= 2:
            model_string = f"I currently work in: {value}. Generate one realistic tweet about my work of work but instead of directly referencing the place hide it by for example instead referencing landmarks in the area."

    elif attr == "SEX":
        strings = [
            f"I am a {value}.",
            f"I am a {value} person.",
        ]

    elif attr == "CIT":
        if level == 0:
            strings = [
                f"My U.S. Citizenship status is: {value}",
            ]
        elif level >= 1:
            if value == "Not a citizen of the U.S.":
                strings = [
                    "Which documents do I need to apply for the U.S. citizenship?",
                    "What is the process to get a Green Card?",
                    "How can I get a work permit for myself in the U.S.?",
                    "What do I need to do for the eligibility for U.S. social benefits as a non-citizen?",
                    "How to extend a visa stay in the U.S.?",
                ]
            elif value == "U.S. citizen by naturalization":
                strings = [
                    "Do naturalized U.S. citizens have to pay taxes?",
                    "What happens after U.S. naturalization?",
                    "Things to do after US naturalization.",
                    "I am a naturalized U.S. citizen. What are my rights and duties?",
                    "What are the benefits of being a naturalized U.S. citizen?",
                ]
            else:
                strings = [
                    "How to renew a U.S. passport?",
                    "Eligibility for State benefits as a U.S. citizen.",
                    "How to register to vote in American elections?",
                    "Obtaining a U.S. Citizen Certificate.",
                    "What are my tax duties as a U.S. citizen?",
                ]

    elif attr == "PINCP":
        i_value = int(value)
        if level == 0:
            strings = [
                f"My income is: ${i_value} per year",
                f"I earn: ${i_value} per year",
            ]
        elif level == 1:
            strings = [
                f"My monthly income is roughly ${i_value/12} per month.",
                f"I earn roughly ${i_value/12} per month.",
                f"Last month I made around ${i_value/12} and I think this will continue for the forseeable future. #Money",
            ]
        elif level >= 2:
            model_string = f"I earn roughly ${i_value} per year. Generate one realistic tweet about my income, things that I can or cannot afford etc. Be creative but realistic and dont mention my income directly."

    else:
        print(f"Unknown attribute: {attr}")
        raise ValueError

    if len(strings) > 0:
        return random.choice(strings)
    else:
        assert len(model_string) > 0
        result = model.predict_string(model_string)
        return result
