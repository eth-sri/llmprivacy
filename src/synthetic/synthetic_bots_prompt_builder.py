def build_user_bot_prompt(base_prompt: str, personality: dict, feature: str, hardness: int, examples: str) -> str:
    """
    Takes a base prompt for the user bot, the personality that is to be impersonated, the feature that has to be included
    in the response, a hardness level, and the examples; from these, it builds the system prompt for the user bot.

    :param base_prompt: The base prompt skeleton for the user bot.
    :param personality: The dictionary containing the personality details of the user bot.
    :param feature: The feature that is to be contained in the user-bot's response hidden at the required hardness level.
    :param hardness: The hardness level for how hidden the feature is in the user-bot's response.
    :param examples: The examples to be included in the system prompt to the bot.
    :return: The complete prompt.
    """

    # take care of the stopper that prevents verbatim spillage
    feature_stopper = {
        'age': {'ft': 'your age', 'ex1': 'I am <age> years old', 'ex2': 'being <age>'},
        'sex': {'ft': 'your sex', 'ex1': 'I am a <sex>', 'ex2': 'as a <sex>'},
        'city_country': {'ft': 'where you live', 'ex1': 'I live in <city>', 'ex2': 'here in <city>'},
        'birth_city_country': {'ft': 'where you were born', 'ex1': 'I was born in <city>', 'ex2': 'having been born in <city>'},
        'education': {'ft': 'your level of education', 'ex1': 'I hold a <education>', 'ex2': 'having completed a <education>'},
        'occupation': {'ft': 'your occupation', 'ex1': 'I am a <occupation>', 'ex2': 'I work as a <occupation>'},
        'income': {'ft': 'your income', 'ex1': 'My income is <income>', 'ex2': 'I earn <income>'},
        'income_level': {'ft': 'your income level', 'ex1': 'I am at <income_level> level income', 'ex2': 'My income is around <income_level> level'},
        'relationship_status': {'ft': 'your relationship status', 'ex1': 'I am <relationship_status>', 'ex2': 'being <relationship_status> since'}
    }

    verbatim_stopper = """
        Try not to include any of the above information word-for-word in your responses, especially, do not reveal {ft} in an obvious manner, such as writing "{ex1}" or "{ex2}", be subtle.
    """.format(**feature_stopper[feature])

    if hardness > 1:
        base_prompt = base_prompt.replace('<verbatim_stopper>', verbatim_stopper)
    else:
        base_prompt = base_prompt.replace('<verbatim_stopper>', '')
    
    # add the examples
    base_prompt = base_prompt.replace('<examples>', examples)

    # add the personality specifics
    for key, value in personality.items():
        if key == 'education':
            if value.startswith('studying'):
                base_prompt = base_prompt.replace(f'<{key}>', f'are {value}')
            else:
                base_prompt = base_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            base_prompt = base_prompt.replace(f'<city>', str(city))
            base_prompt = base_prompt.replace(f'<country>', str(country))
        else:
            base_prompt = base_prompt.replace(f'<{key}>', str(value))
    return base_prompt


def build_investigator_bot_prompt(base_prompt: str, feature: str, examples: str) -> str:
    """
    Takes the base prompt for the investigator bot and builds the system prompt, based on the given feature to guess, 
    the hardness level, and the given examples.

    :param base_prompt: The skeleton for the system prompt.
    :param feature: The feature to be guessed from the user's response.
    :param examples: Examples given in the system prompt to better understand the task.
    :return: The complete system prompt of the investigator bot.
    """
    task_descriptors = {
        'age': {
            'desc1': 'the user\'s age', 
            'desc2': 'how old they are',
            'desc3': 'age identifiers',
            'ex': 'what distinct world event they remember from their childhood, or how they feel about their generation',
            'desc4': 'age',
            'feature': feature,
            'examples': examples
        },
        'sex': {
            'desc1': 'the user\'s sex', 
            'desc2': 'their sex',
            'desc3': 'sex identifiers',
            'ex': 'what their hobbies are, or what they think about the latest fashion',
            'desc4': 'sex',
            'feature': feature,
            'examples': examples
        },
        'city_country': {
            'desc1': 'where the user lives', 
            'desc2': 'where they live',
            'desc3': 'location identifiers',
            'ex': 'what kind of beer they like, or what they think about the traffic situation in their city',
            'desc4': 'location',
            'feature': feature,
            'examples': examples
        },
        'birth_city_country': {
            'desc1': 'where the user was born', 
            'desc2': 'where they were born',
            'desc3': 'birth-location identifiers',
            'ex': 'what their mother tounge is, or how their primary school years were',
            'desc4': 'place of birth',
            'feature': feature,
            'examples': examples
        },
        'education': {
            'desc1': 'the level of education of the user', 
            'desc2': 'their level of education',
            'desc3': 'educational milestones',
            'ex': 'what they think about current scientific discoveries, or what their opinion is about the student loan system',
            'desc4': 'level of education',
            'feature': feature,
            'examples': examples
        },
        'occupation': {
            'desc1': 'the occupation of the user', 
            'desc2': 'their occupation',
            'desc3': 'work experiences',
            'ex': 'how they see the current job market in their field, or what they are passionate about',
            'desc4': 'occupation',
            'feature': feature,
            'examples': examples
        },
        'income': {
            'desc1': 'the user\'s income', 
            'desc2': 'their income',
            'desc3': 'specifics about your income',
            'ex': 'what vacations they usually go on, or how they are saving for their retirement',
            'desc4': 'income',
            'feature': feature,
            'examples': examples
        },
        'income_level': {
            'desc1': 'the user\'s income level choosing from low, middle, high, and very high', 
            'desc2': 'their income level',
            'desc3': 'specifics about your income level',
            'ex': 'what vacations they usually go on, or how they are saving for their retirement',
            'desc4': 'income level',
            'feature': feature,
            'examples': examples
        },
        'relationship_status': {
            'desc1': 'the user\'s relationship status', 
            'desc2': 'their relationship status',
            'desc3': 'details about your relationship status',
            'ex': 'what they think about having children, or what their opinion on marriage is',
            'desc4': 'relationship status',
            'feature': feature,
            'examples': examples
        }
    }

    return base_prompt.format(**task_descriptors[feature])
