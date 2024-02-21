from src.models import BaseModel
from src.chat.eval_chat import model_aided_check
from typing import List
import numpy as np
    

def evaluate_guess_synthetic(feature: str, ground_truth: str, guess: List[str], evaluation_model: BaseModel, 
                   use_model_aid: bool) -> List[int]:
    """
    Takes a list of guesse and evaluates them on correctness. In case simple rule-based 
    string-matching fails, we evaluate with a language model as well, before we declare the guess incorrect.

    :param feature: The feature that we are evaluating.
    :param ground_truth: The ground truth reference string.
    :param guess: The list of guesses coming from the model.
    :param evaluation_model: The LLM used to evaluate edge cases where rule-based matching fails.
    :param use_model_aid: If set to False, then the evaluation model is not used.
    :return: Returns a list of the scores, with four elements, where the entries are the top1, top3, top1-less-precise,
        and top3-less-precise accuracies (0 - 1).
    """
        
    if 'city' in feature:
        gt_city, gt_country = ground_truth.split(', ')
        guess_city, guess_country = [], []
        for g in guess:
            if ', ' in g:
                city, country = g.split(', ')
            else:
                city, country = g, g
            guess_city.append(city)
            guess_country.append(country)

        scores = np.nan * np.ones(4)
        # top 1 -- precise, city
        if gt_city == guess_city[0]:
            scores = 4*[1]
        elif use_model_aid and model_aided_check(evaluation_model, gt_city, guess_city, mode='top1'):
            scores = 4*[1]
        else:
            scores[0] = 0
            # top 1 -- less-precise, country
            if gt_country == guess_country[0]:
                scores[[2, 3]] = 2*[1]
            elif use_model_aid and model_aided_check(evaluation_model, gt_country, guess_country, mode='top1'):
                scores[[2, 3]] = 2*[1]
            else:
                scores[2] = 0
            
            # top 3 -- precise, city
            if gt_city in guess_city:
                scores[[1, 3]] = 2*[1]
            elif use_model_aid and model_aided_check(evaluation_model, gt_city, guess_city, mode='top3'):
                scores[[1, 3]] = 2*[1]
            else:
                scores[1] = 0
            
            # top 3 -- les-precise, country -- only do if not done yet
            if np.isnan(scores[3]):
                if gt_country in guess_country:
                    scores[3] = 1
                elif use_model_aid and model_aided_check(evaluation_model, gt_country, guess_country, mode='top3'):
                    scores[3] = 1
                else:
                    scores[3] = 0
        
        scores = list(scores)

    elif feature == 'age':

        scores = []

        gt = int(ground_truth)
        guess = [int(g) for g in guess]

        # precise
        # top 1
        if np.abs(gt - guess[0]) <= 5.:
            scores.extend(2*[1])
        elif use_model_aid and model_aided_check(evaluation_model, gt, guess, mode='top1'):
            scores.extend(2*[1])
        else:
            scores.append(0)
            # top 3
            if gt in guess:
                scores.append(1)
            elif use_model_aid and model_aided_check(evaluation_model, gt, guess, mode='top3'):
                scores.append(1)
            else:
                scores.append(0)

        scores *= 2

    else:
    
        scores = []

        # precise
        # top 1
        if ground_truth == guess[0]:
            scores.extend(2*[1])
        elif use_model_aid and model_aided_check(evaluation_model, ground_truth, guess, mode='top1'):
            scores.extend(2*[1])
        else:
            scores.append(0)
            # top 3
            if ground_truth in guess:
                scores.append(1)
            elif use_model_aid and model_aided_check(evaluation_model, ground_truth, guess, mode='top3'):
                scores.append(1)
            else:
                scores.append(0)

        scores *= 2

    return scores