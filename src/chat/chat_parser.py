import re
import json
from typing import List


def check_if_guess(response: str) -> bool:
    """
    Uses regular expressions to find if the response contains the required inference pattern. If yes, that means
    that this is a guess response, and we return True. Otherwise, we return False.

    :param response: The response to be checked.
    :return: True if the response contains at least one match for the inference pattern.
    """
    # 1. pattern = r"Type: (.*?)(\n)?Inference: (.*?)(\n)?Guess: (.*?); (.*?); (.*)"
    pattern = r"Type: ((.|\n)*?)Inference: ((.|\n)*?)Guess:\n?\n?([^;]*); ([^;]*); ([^;]*)"
    match = re.search(pattern, response, re.DOTALL)
    return match is not None
    

def extract_predictions(guess: str) -> dict[str, List[str]]:
    """
    Takes the final guess response of the model and interprets it into a representation that can be matched to calculate the accuracy.
    
    :param guess: The guess response of the investogator model.
    :return: A dictionary with keys naming the feature and items being lists of strings for the first three guesses.
    """
    assert check_if_guess(guess), f'\"{guess}\"\nIs not a guess.'
    result = {}
    split_guess = guess.replace('Type:', '<split_id>Type:').split('<split_id>')[1:]
    pattern = r"Type: ((.|\n)*?)Inference: ((.|\n)*?)Guess:\n?\n?([^;]*);([^;]*);([^;]*)"

    for sg in split_guess:
        matches = re.findall(pattern, sg, re.DOTALL)
        for match in matches:
            type_key = match[0].strip()
            guess_values = [match[4].strip(), match[5].strip(), match[6].strip().replace('.', '')]
            result[type_key] = guess_values

    return result


def extract_chat(chat: str) -> dict:
    """
    Takes a full chat and creates a list of 'role' 'response' pairs.

    :param chat: The unparsed chat.
    :return: The parsed chat in the above form.
    """
    split_chat = chat.split('Conversation Start')[1].strip()
    if split_chat.startswith('INVESTIGATOR'):
        starter = 'INVESTIGATOR'
        answerer = 'USER'
    elif split_chat.startswith('USER'):
        starter = 'USER'
        answerer = 'INVESTIGATOR'
    else:
        raise ValueError('Invalid chat format. The first conversation element does not start with either USER or INVESTIGATOR.')
    split_chat = split_chat.split('INVESTIGATOR:')
    split_chat = [c.split('USER:') for c in split_chat]
    split_chat = [sc.strip() for subchat in split_chat for sc in subchat if sc != '']

    return [(starter, c) if idx%2 == 0 else (answerer, c) for idx, c in enumerate(split_chat)]


def parse_chat(chat: str) -> dict:
    """
    takes a single chat, and parses it into a dictionary of the structure:
    {
    'personality': personality_dict,
    'chat': list(('role', 'response'))
    'prediction': {
            'feature': ['first_guess', 'second_guess', 'third_guess']
        } 
    }

    :param chat: The raw chat between the investigator bot and the user bot.
    :return: The dictionary of the parsed chat in the format specified above.
    """
    parsed_chat = {}
    parsed_chat['personality'] = json.loads(re.search(r'\{.*?\}', chat).group().replace('\'', '\"'))
    parsed_chat['chat'] = extract_chat(chat)
    parsed_chat['prediction'] = extract_predictions(parsed_chat['chat'][-1][-1])

    return parsed_chat


def parse_chats(chats: str) -> List[dict]:
    """
    Splits the chats over personalities and parses each of the single personality chats into a dictionary 
    ready for evaluation.

    :param chat: The raw chat between the investigator bot and the user bot.
    :return: List of the parsed chats.
    """
    # slice at each chat
    chats_sliced = re.split(r'# ------ Conversation with pers\d+ ------ #', chats)[1:]
    return [parse_chat(chat) if 'An exception occurred' not in chat else None for chat in chats_sliced]


def parse_investigator_response(response: str) -> tuple[str]:
    """
    In case of chat with reasoning, the investigator's response is split into current knowledge, the
    response intended to the user, and the reasoning for the type of response given. The structure of the response has to be:
    
    What I know already: <text>
    My response to the user: <text>
    Reasoning for my response: <text>

    :param response: The whole, unformatted response of the investigator bot.
    :return: The response split into the three parts elaborated above.
    """
    knowledge, remaining_response = response.split('My response to the user:')
    knowledge = knowledge.replace('What I know already:', '')
    response_to_user, reasoning = remaining_response.split('Reasoning for my response:')

    return knowledge.strip(), response_to_user.strip(), reasoning.strip()
