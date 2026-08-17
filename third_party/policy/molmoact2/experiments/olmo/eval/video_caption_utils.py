import json
import re
import time
from typing import Dict
from tqdm import tqdm
import openai


def get_chat_response(
        prompt,
        api_key,
        model="gpt-4-0613",
        temperature=0,
        max_tokens=256,
        n=1,
        patience=10000000,
        sleep_time=0,
        system_prompt=None,
        **kwargs
):
    """Run a query through an OpenAI model"""

    messages = [
        {"role": "user", "content": prompt},
    ]
    if system_prompt is not None:
        messages = [
                       {"role": "system", "content": system_prompt}
                   ] + messages

    client = openai.OpenAI(
        api_key=api_key,
    )
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                **kwargs
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce prompt size")
                # reduce input prompt and keep the tail
                new_size = int(len(prompt) * 0.9)
                new_start = len(prompt) - new_size
                prompt = prompt[new_start:]
                messages = [
                    {"role": "user", "content": prompt},
                ]

            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def query_gpt(prompt: str, openai_api_key: str, system_prompt: str = None, maxtry: int = 10):
    if system_prompt is None:
        system_prompt = "You are an AI assistant for question answering."
    gen_params = dict(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        top_p=0.1,
        max_tokens=10240,
        presence_penalty=1,
        patience=maxtry,
    )
    llm_output = get_chat_response(prompt, openai_api_key, system_prompt=system_prompt, **gen_params)
    return llm_output


def get_canonical_statements(model_caption: str, openai_api_key: str):
    canonical_statements_prompt = f"""
    Based on the description of the video, come up with a list of the MOST canonical statements that are mentioned in it. 
    Each statement should be self-contained and broken down as much as possible.
    The statements should be an ordered list, where each item is separated a newline. For instance, the response may look like:\n\n1. Statement A\n2. Statement B\n3. Statement C\n\n\n"

    Here is the video description: {model_caption}"
    """
    return query_gpt(
        canonical_statements_prompt,
        openai_api_key,
        system_prompt="You are an AI assistant for generating canonical statements from video descriptions.",
    )


def get_consistency_statements(
        gt_caption: str, statements_str: str, openai_api_key: str
):
    prompt = (
            f"Here is a caption of a video.\n\n"
            + (
                # captions
                gt_caption
            )
            + (
                '\n\nHere are statements that a captioning model made about the video. For each statement, state whether it\'s "Consistent" or "Inconsistent" with the captions provided above. The output should be in the form\n\n1. Consistent\n2. Inconsistent\n3. Consistent\n\nDo not output anything other than an ordered list of Consistent and Inconsistent.\n\n'
            )
            + (
                # statements
                statements_str
            )
    )
    return query_gpt(prompt, openai_api_key=openai_api_key, system_prompt="You are an AI assistant for evaluating caption consistency.")


def eval_caption_consistency(
        prediction: str,
        data: Dict,
        openai_api_key: str,
) -> float:
    statements_str = get_canonical_statements(prediction, openai_api_key)
    gt_caption = data['caption']
    consistency_statements = get_consistency_statements(gt_caption, statements_str, openai_api_key)
    lines = [x.strip() for x in consistency_statements.split("\n") if x.strip()]
    scores = []
    for line in lines:
        inconsistent = None
        # GPT 4 is surprisingly bad at following in consistent/inconsistent format exactly,
        # do some fuzzy matching for mispellings and other variations
        if re.fullmatch(r".*[^a-z]((i?inconsis?ten(t|cy)?)|incorrect|inconsistence|iconsistent|inconsisent|incomplete|contradictory).*", line, flags=re.IGNORECASE):
            inconsistent = True

        if re.fullmatch(r".*[^a-z](consistent(ly)?|constistent|correct).*$", line, flags=re.IGNORECASE):
            if inconsistent:
                inconsistent = None
            else:
                inconsistent = False

        scores.append(inconsistent)
        # if inconsistent is None:
        #     statement_errors.append(f"Bad consistency output {line}")
        #     # Model is not instructed to output these unknown options, but does anyway
        #     unknown = [
        #         "not specified",
        #         "cannot determine",
        #         "not determinable",
        #         "no verification",
        #         "N/A",
        #         "not confirmed",
        #         "neither",
        #         "not stated",
        #         "no judgement",
        #         "unable to determine",
        #         "inconclusive",
        #         "undetermined",
        #         "insufficient information",
        #         "no relevant information",
        #         "no conclusion",
        #         "not clear",
        #         "unknown",
        #         "uncertain",
        #         "ambiguous",
        #         "not addressed",
        #         "not enough information",
        #         "not mentioned",
        #         "not enough info",
        #         "no information",
        #         "not verifiable",
        #         "not applicable"
        #     ]
        #     if not re.fullmatch(r".*\b(" + "|".join(unknown) + r").*$", line, flags=re.IGNORECASE):
        #         # Warn if it is something very unexpected
        #         logging.warning(statement_errors[-1])
    scores = [x for x in scores if x is not None]
    return 1 - float(sum(scores) / len(scores)) if len(scores) > 0 else 0


def get_recall_statements(
        gt_statements: str, caption_str: str, openai_api_key: str
):
    prompt = (
            f"Here are statements about a video.\n\n"
            + (
                # captions
                gt_statements.strip()
            )
            + (
                '\n\nNext, consider the following caption of the video. For each statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\nDo not output anything other than an ordered list of Stated and Not Stated.\n\n Here is the caption: '
            )
            + (
                # statements
                caption_str.strip()
                if caption_str
                else "No caption provided."
            )
    )
    return query_gpt(prompt, openai_api_key=openai_api_key, system_prompt="You are an AI assistant for evaluating caption recall.")


def eval_caption_recall(
        prediction: str,
        data: Dict,
        openai_api_key: str,
) -> float:
    recall_statements = get_recall_statements(data['statements'], prediction, openai_api_key)
    lines = [x.strip() for x in recall_statements.split("\n") if x.strip()]
    scores = []
    for line in lines:
        valid = None
        # GPT is mispells "not stated" sometimes, give it some slack
        if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
            valid = False
        elif " stated" in line.lower():
            valid = True
        scores.append(valid)
    scores = [x for x in scores if x is not None]
    return float(sum(scores) / len(scores)) if len(scores) > 0 else 0


def eval_caption(
        prediction: str,
        data: Dict,
        openai_api_key: str,
) -> Dict:
    """
    Evaluate caption using OpenAI API.
    :param prediction: predicted caption
    :param data: ground truth data
    :param openai_api_key: OpenAI API key
    :return: evaluation results
    """
    recall = eval_caption_recall(prediction, data, openai_api_key)
    consistency = eval_caption_consistency(prediction, data, openai_api_key)
    return {
        "recall"     : recall,
        "consistency": consistency,
    }


