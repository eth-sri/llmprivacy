def build_user_bot_system_prompt(prompt_skeleton: str, profile: dict) -> str:
    """
    Takes a parameterized prompt skeleton for the user bot and a profile dictionary
    and return the filled out prompt skeleton, ready to use for the bot.

    :param prompt_skeleton: A string with parameters <param> matching the keys in the dictionary,
        This skeleton is filled out to generate the system prompt.
    :param profile: A dictionary containing the personal information of the profile that is to be
        impersonated by the user bot.
    :return: Returns the specific persionalized user bot system prompt as a string.
    """
    final_prompt = prompt_skeleton
    for key, value in profile.items():
        if key == 'education':
            if value.startswith('studying'):
                final_prompt = final_prompt.replace(f'<{key}>', f'are {value}')
            else:
                final_prompt = final_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            final_prompt = final_prompt.replace(f'<city>', str(city))
            final_prompt = final_prompt.replace(f'<country>', str(country))
        else:
            final_prompt = final_prompt.replace(f'<{key}>', str(value))
    return final_prompt
