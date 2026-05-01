'''
This utility file contains the functions to calculate the metrics for the metaphoricity rating task, the second stage of the open-ended metaphor detection task
'''
import re
from operator import itemgetter
import ast
import math
import json

def extract_dict_candidates(s: str) -> list[str]:
    stack = []
    spans = []

    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append(s[start:i+1])

    #shortest first, most dictionary object and not JSON
    spans.sort(key= len, reverse= False)
    return spans

def parse_response(dict_str: str) -> dict | float:
    if not isinstance(dict_str, str):
        return float("NaN")

    #direct literal parse
    try:
        parsed = ast.literal_eval(dict_str)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass

    #try candidates based on brace pushing and popping
    for candidate in extract_dict_candidates(dict_str):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass

        try:
            cleaned = candidate
            cleaned = cleaned.replace("null", "None")
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
    return float("NaN")

def create_human_ratings(annotations: list[str], scores: list[float]) -> dict:
    assert(len(annotations) == len(scores))

    ratings = {-1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(0, len(annotations)):
        annotation = annotations[i]
        score = int(scores[i])
        ratings[score].append(annotation)
    return ratings

def get_human_ratings(human_ratings: dict) -> list[float]:
    human_keys = list(human_ratings.keys())
    if(human_keys == []):
        return float("NaN")
    human_values_for_human_ratings = list(itemgetter(*human_keys)(human_ratings))
    return human_values_for_human_ratings

def filter_human_ratings(human_ratings: dict) -> dict:
    new_ratings = {}
    for key, value in human_ratings.items():
        if(key != -1):
            new_ratings[key] = value
    return new_ratings

def get_llm_ratings_of_overlapped_samples(human_ratings: dict[int, list[str]], llm_predictions: dict[int, list[str]]) -> dict[int, list[str]]:
    #flatten human ratings into a global set of strings
    human_global_set = {
        item
        for items in human_ratings.values()
        for item in items
    }

    overlapped = {}

    for llm_rating, llm_items in llm_predictions.items():
        overlap = [x for x in llm_items if x in human_global_set]
        if overlap:
            overlapped[llm_rating] = overlap
    return overlapped

def get_llm_ratings_of_overlapped_samples_(human_ratings: dict[int, list[str]], llm_predictions: dict[int, list[str]]) -> dict[int, list[str]]:
    #flatten human ratings into a global set of strings
    human_global_set = {
        item
        for items in human_ratings.values()
        for item in items
    }

    overlapped = {0: [], 1: [], 2: [], 3: []}

    for llm_rating, llm_items in llm_predictions.items():
        overlap = [x for x in llm_items if x in human_global_set]
        if overlap:
            overlapped[int(llm_rating)].append(overlap)
    return overlapped

def get_llm_ratings_of_llm_only_samples(
    human_ratings: dict[int, list[str]],
    llm_predictions: dict[int, list[str]]
) -> dict[int, list[str]]:
    # flatten all human-rated words into a global set
    human_global_set = {item for items in human_ratings.values() for item in items}

    llm_only = {}

    for llm_rating, llm_items in llm_predictions.items():
        # keep only words NOT in human annotations
        unique_items = [x for x in llm_items if x not in human_global_set]
        if unique_items:
            llm_only[llm_rating] = unique_items

    return llm_only


def self_inconsistancy_score(
    human_ratings: dict[int, list[str]],
    llm_predictions: dict[int, list[str]]
) -> int:
    # get LLM-only words (i.e., words not present in human annotations)
    llm_only = get_llm_ratings_of_llm_only_samples(human_ratings, llm_predictions)

    # safely get words rated 0 (literal) by LLM among the LLM-only words
    zeros = llm_only.get(0, [])

    return len(zeros)

def self_inconsistancy_bool(llm_predictions: dict, llm_metaphor_list: list[str]) -> bool:
    score = self_inconsistancy_score(llm_predictions, llm_metaphor_list)
    if(isinstance(score, float) and math.isnan(score)):
        return float("NaN")
    return score > 0

'''Counts how often the model contradicts the human annotation returns in following order: 
    - Number where llm rating is above human rating
    - Number where llm rating is below human rating
    - Number where llm rating is equal to human rating
    - Number where llm rating is literal while human rating is metaphorical
    - Number where llm rating is metaphorical while human rating is literal'''

def human_contradiction_scores(human_ratings: dict, llm_predictions: dict) -> int:
    human_ratings = filter_human_ratings(human_ratings)
    if(human_ratings == {}): #only invalid annotations
        return float("NaN")
    overlapped_ratings = get_llm_ratings_of_overlapped_samples(human_ratings, llm_predictions)

    human_ratings = tuple(human_ratings.items())
    overlapped_ratings = tuple(overlapped_ratings.items())

    #explode lists, remove empty lists
    human_ratings = tuple(
        (int(k), v_item)
        for k, v in human_ratings
        for v_item in v
    )

    overlapped_ratings = tuple(
        (int(k), v_item)
        for k, v in overlapped_ratings
        for v_item in v
    )

    #sort by second value alphabetically decending in this list of 2-tuples
    human_ratings = tuple(sorted(human_ratings, key=lambda x: x[1], reverse= True))
    overlapped_ratings = tuple(sorted(overlapped_ratings, key=lambda x: x[1], reverse= True))

    if(len(human_ratings) != len(overlapped_ratings)): #either llm made a mistake or more than one annotation
        return float("NaN")

    more_count = 0
    less_count = 0
    equal_count = 0
    literal_count = 0
    metaphorical_count = 0
    for i in range(0, len(overlapped_ratings)):
        assert(overlapped_ratings[i][1] == human_ratings[i][1])
        llm_score = overlapped_ratings[i][0]
        human_score = human_ratings[i][0]
        if(llm_score > human_score):
            more_count += 1
            if(llm_score > 0 and human_score == 0):
                metaphorical_count += 1
        elif(llm_score < human_score):
            less_count += 1
            if(llm_score == 0 and human_score > 0):
                literal_count += 1
        else:
            equal_count += 1
    return more_count, less_count, equal_count, literal_count, metaphorical_count

def human_contradiction_scores_bool(human_ratings: dict, llm_predictions: dict) -> list[bool]:
    results = human_contradiction_scores(human_ratings, llm_predictions)
    if(isinstance(results, float)):
        return float("NaN")
    else:
        return [result > 0 for result in list(results)]



