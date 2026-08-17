import os
import random
import re
import json
import pandas as pd
from openai import OpenAI

NUM_FRAME = '16'

# number of maximum re-try in parsing
MAX_ITER = 5


def gpt_parser(response, all_choices, index2ans):
    print("using gpt parser...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""You are given a response, a list of multiple-choice options, and a index2answer mapping. You are required to extract the letter option from the GPT. 
    
    response: {response}

    all_choices: {all_choices}

    index2answer: {index2ans}

Only output the single parsed letter from the response. No other texts are needed. 

If you think no options can match the index2answer dictionary, randomly select one letter. 

Your extracted letter is: 
"""
    prompt_message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    params = {
        "model": "gpt-4o-mini",
        "messages": prompt_message,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    response = client.chat.completions.create(**params)
    response = response.choices[0].message.content

    return response


def pre_parser(response, all_choices, index2ans):
    parsed_response = ""
    response = response.strip()

    # preprocess matches
    full_choices = [f'{k}: {v}' for k, v in index2ans.items()]
    pattern = r"^Answer is:?[\(]?([A-Fa-f])[\)]?$"
    match = re.match(pattern, response)

    # exact match single letter
    if len(response) == 1 and response.upper() in all_choices:
        parsed_response = response.upper()

    # exact match of the choice
    elif response.upper() in full_choices:
        parsed_response = response[0].upper()

    # regex match of "Answer is: A", "Answer is (A)", etc
    elif match:
        parsed_response = match.group(1).upper()

    return parsed_response

def parse_result(id_, question, response, all_choices, index2ans, gt):
    parsed_response = pre_parser(response=response,
                                 all_choices=all_choices,
                                 index2ans=index2ans)

    # actual parsing using gpt
    if parsed_response not in all_choices:
        curr_iter = 0
        while curr_iter < MAX_ITER:
            response_candidate = gpt_parser(response=response,
                                            all_choices=all_choices,
                                            index2ans=index2ans)

            if response_candidate in all_choices:
                parsed_response = response_candidate
                break
            curr_iter += 1

    if parsed_response not in all_choices:
        parsed_response = random.choice(all_choices)

    # format parsed result
    parsed_result = {
        "id": id_,
        "question": question,
        "response": parsed_response,
        "gt": gt
    }

    return parsed_result


def get_single_score(result):
    response = result['response']
    gt = result['gt'][0]
    return response == gt


def get_tomato_score(response, metadata):
    result = parse_result(metadata["id"], metadata["question"], metadata["all_choices"], metadata["answer"])
    return get_single_score(result)
