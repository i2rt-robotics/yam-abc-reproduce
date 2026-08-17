import difflib
import json
import random
import re
import time
from collections import defaultdict
from typing import Dict

import openai
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def get_chat_response(
        prompt,
        api_key,
        model="gpt-4-0613",
        temperature=0,
        n=1,
        patience=10000000,
        sleep_time=10,
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
            # if "Rate limit" not in str(e):
            log.warning(e)

            if "Please reduce the length of the messages" in str(e):
                log.warning("!!Reduce prompt size")
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


def query_gpt(prompt: str, openai_api_key: str, system_prompt: str = None, maxtry: int = 100):
    if system_prompt is None:
        system_prompt = "You are an AI assistant for question answering."
    gen_params = dict(
        # model="o3",
        model="gpt-4.1-2025-04-14",
        temperature=0,
        # top_p=0.1,
        # max_tokens=10240,
        # presence_penalty=1,
        patience=maxtry,
    )
    llm_output = get_chat_response(prompt, openai_api_key, system_prompt=system_prompt, **gen_params)
    return llm_output


def get_canonical_statements(model_caption: str, openai_api_key: str):
    canonical_statements_prompt = f"""
    Based on the description of the video, come up with a list of the MOST canonical statements that are mentioned in it. 
    Each statement should be self-contained and broken down as much as possible.
    The statements should be an ordered list, where each item is separated a newline. For instance, the response may look like:\n\n1. Statement A\n2. Statement B\n3. Statement C\n\n
    Here is the video description:\n\n{model_caption}
    """
    raw_statement = query_gpt(
        canonical_statements_prompt,
        openai_api_key,
        system_prompt="You are an AI assistant for generating canonical statements from video descriptions.",
    )

    split_categorize_statement_prompt = """You are a Video‑Statement Splitter.
    You will be given a list of statements, each seperated by a new line describing a video’s content.
    Your job is to split every statement into multiple concise, atomic statements that capture distinct facts or observations.
    For each output atomic statement, choose exactly one category from the list below.
y
    **Statement Categories**  
    Here’s the list of categories:
    - Object: Concrete entities in the scene (e.g. “dog”, “car”, “tree”)
    - Action: Verbs or activities (e.g. “running”, “kicking”, “talking”)
    - Attribute: Properties of objects or actors
    - Relation: How two or more entities relate
    - Location: Place names or spatial descriptors (e.g. “in the park”, “on the table”)
    - Quantity/Number: Counts or measurements (e.g. “three people”, “2 liters”)
    - State/Condition: Static or changing states (e.g. “door is open”, “water boiling”)
    - Event: Higher‑level happenings (e.g. “birthday party”, “earthquake”)
    - Motion/Trajectory: Movement specifics (e.g. “rolling down”, “flying upward”)
    - Pose: Body configurations (e.g. “sitting”, “arms crossed”)
    - Gesture: Hand or head motions conveying meaning (e.g. “waving”, “nodding”)
    - Emotion/Affect: Inferred feelings (e.g. “smiling happily”, “looks angry”)
    - Identity: Recognized person/place/brand (e.g. “Barack Obama”, “Eiffel Tower”)
    - OCR: Visible textual content with the text explicitly described (e.g. “STOP” sign, "Subscribe" button
    - Camera: Technical/cinematic cues
    - Lighting/Weather: Environmental conditions (e.g. “sunny”, “rainy”, “dimly lit”)
    - Scene/Context: Overall setting or scenario (e.g. “kitchen”, “office meeting”)
    - Causation/Purpose: Cause–effect or intent (e.g. “so that”, “in order to”)

    **Output Requirement**  
    The output should be an ordered list, where each item is '<atomic statement> | <category>' and separated by a newline. 
    """

    category_options = [
        "Object",
        "Action",
        "Attribute",
        "Relation",
        "Location",
        "Quantity/Number",
        "State/Condition",
        "Event",
        "Motion/Trajectory",
        "Pose",
        "Gesture",
        "Emotion/Affect",
        "Identity",
        "OCR",
        "Camera",
        "Lighting/Weather",
        "Scene/Context",
        "Causation/Purpose"
    ]

    atomic_statements = query_gpt(
        raw_statement,
        openai_api_key,
        system_prompt=split_categorize_statement_prompt,
    )

    statements_list, categories = [], []
    for line in atomic_statements.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Split the statement and category
        try:
            s, c = line.rsplit("|", maxsplit=1)
        except ValueError as e:
            log.warning(f'Warning: [generate statement] output from model: {line}')
            s = line
            c = random.choice(category_options)
        s = s.strip()
        c = c.strip()
        statements_list.append(s)
        if c not in category_options:
            c = difflib.get_close_matches(c, category_options, n=1, cutoff=0)[0]
        categories.append(c)
    return statements_list, categories


def reindex_list(text):
    """
    Reindexes numbered lines in a string to start from 1.

    Args:
        text (str): Multiline string with lines starting with a number and a period.

    Returns:
        str: The text with lines renumbered starting from 1.
    """
    lines = text.splitlines()
    new_lines = []
    for idx, line in enumerate(lines, start=1):
        # Replace the leading number (one or more digits) followed by a period with the new index.
        new_line = re.sub(r'^\d+\.', f'{idx}.', line)
        new_lines.append(new_line)
    return '\n'.join(new_lines)


def get_consistency_statements(
        gt_caption: str, statements_str: str, openai_api_key: str
):
    statements_str = reindex_list(statements_str)
    prompt = (
            f"Here are several description sources of th same video: time‑stamped, clip‑level human captions (as the authoritative source) and one model‑generated caption.\n\n"
            + (
                # captions
                gt_caption
            )
            + (
                '\n\n#####\n\n'
            )
            + (
                'Here are statements that a captioning model made about the video. For each statement, state whether it\'s "Consistent" or "Inconsistent" with the captions provided above. The output should be in the form\n\n1. Consistent\n2. Inconsistent\n3. Consistent\n\nDo not output anything other than an ordered list of Consistent and Inconsistent.\n\n'
            )
            + (
                '##### Statements:\n\n'
            )
            + (
                # statements
                statements_str
            )
    )
    return query_gpt(prompt, openai_api_key=openai_api_key, system_prompt="You are an AI assistant for evaluating caption consistency.")


def eval_caption_consistency(
        statements_list,
        category_list,
        data: Dict,
        openai_api_key: str,
        batch_size: int = -1,
):
    gt_caption = data['aggregated_annotations']

    scores = []
    category_to_scores_list = defaultdict(list)
    batch_size = batch_size if batch_size > 0 else len(statements_list)
    all_const_statements = []
    for i in range(0, len(statements_list), batch_size):
        n = len(statements_list[i:i + batch_size])
        batch_statements = '\n'.join(statements_list[i:i + batch_size])
        categories = category_list[i:i + batch_size]
        consistency_statements = get_recall_statements(batch_statements, gt_caption, openai_api_key)
        lines = [x.strip() for x in consistency_statements.split("\n") if x.strip()]
        for ii in range(min(n, len(lines))):
            line = lines[ii]
            valid = None
            # GPT is mispells "not stated" sometimes, give it some slack
            if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
                valid = False
            elif " stated" in line.lower():
                valid = True
            if valid is None:
                log.warning(f'Warning: [consistency] output from model: {line}')
            full_statement = f"{categories[ii]} | {statements_list[i:i + batch_size][ii]} | {valid}"
            scores.append(valid)
            category_to_scores_list[categories[ii]].append(valid)
            all_const_statements.append(full_statement)


    scores = [x for x in scores if x is not None]
    category_to_scores = {}
    for category, category_scores in category_to_scores_list.items():
        category_scores = [x for x in category_scores if x is not None]
        category_to_scores[category] = float(sum(category_scores) / len(category_scores)) if len(category_scores) > 0 else 0
    return float(sum(scores) / len(scores)) if len(scores) > 0 else 0, category_to_scores, all_const_statements


def get_recall_statements(
        gt_statements: str, caption_str: str, openai_api_key: str
):
    gt_statements = reindex_list(gt_statements)
    prompt = (
            f"Here are statements about a video.\n\n"
            + (
                # captions
                gt_statements.strip()
            )
            + (
                '\n\n#####\n\n'
            )
            + (
                'Next, consider the following caption of the video. For each statement above, state whether the fact is "Stated" or "Not Stated" in the caption. The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\nDo not output anything other than an ordered list of Stated and Not Stated.\n\n Here is the caption:\n\n'
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
        batch_size: int = -1
):
    statements_list = data['atomic_statements']
    category_list = data['categories']
    scores = []
    category_to_scores_list = defaultdict(list)
    all_recall_statements = []
    batch_size = batch_size if batch_size > 0 else len(statements_list)
    for i in range(0, len(statements_list), batch_size):
        n = len(statements_list[i:i + batch_size])
        batch_statements = '\n'.join(statements_list[i:i + batch_size])
        categories = category_list[i:i + batch_size]
        recall_statements = get_recall_statements(batch_statements, prediction, openai_api_key)
        lines = [x.strip() for x in recall_statements.split("\n") if x.strip()]
        for ii in range(min(n, len(lines))):
            line = lines[ii]
            valid = None
            # GPT is mispells "not stated" sometimes, give it some slack
            if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
                valid = False
            elif " stated" in line.lower():
                valid = True
            if valid is None:
                log.warning(f'Warning: [recall] output from model: {line}')
            full_statement = f"{categories[ii]} | {statements_list[i:i + batch_size][ii]} | {valid}"
            scores.append(valid)
            category_to_scores_list[categories[ii]].append(valid)
            all_recall_statements.append(full_statement)

    scores = [x for x in scores if x is not None]
    category_to_scores = {}
    for category, category_scores in category_to_scores_list.items():
        category_scores = [x for x in category_scores if x is not None]
        category_to_scores[category] = float(sum(category_scores) / len(category_scores)) if len(category_scores) > 0 else 0
    return float(sum(scores) / len(scores)) if len(scores) > 0 else 0, category_to_scores, all_recall_statements


def eval_vixmo_caption(
        example_idx: int,
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
    statements_list, category_list = get_canonical_statements(prediction, openai_api_key)

    data_w_gemini = data['data_with_gemini']
    recall_w_gemini, category_to_recall_w_gemini, recall_statements_w_gemini = eval_caption_recall(prediction, data_w_gemini, openai_api_key)
    consistency_w_gemini, category_to_consistency_w_gemini, consistency_statements_w_gemini = eval_caption_consistency(statements_list, category_list, data_w_gemini, openai_api_key)

    data_wo_gemini = data['data_without_gemini']
    recall_wo_gemini, category_to_recall_wo_gemini, recall_statements_wo_gemini = eval_caption_recall(prediction, data_wo_gemini, openai_api_key)
    consistency_wo_gemini, category_to_consistency_wo_gemini, consistency_statements_wo_gemini = eval_caption_consistency(statements_list, category_list, data_wo_gemini, openai_api_key)


    return {
        "example_idx": example_idx,
        "recall_w_gemini"     : recall_w_gemini,
        "category_recall_w_gemini": category_to_recall_w_gemini,
        "consistency_w_gemini": consistency_w_gemini,
        "category_consistency_w_gemini": category_to_consistency_w_gemini,
        "recall_statements_w_gemini": recall_statements_w_gemini,
        "consistency_statements_w_gemini": consistency_statements_w_gemini,
        "recall_wo_gemini"     : recall_wo_gemini,
        "category_recall_wo_gemini": category_to_recall_wo_gemini,
        "consistency_wo_gemini": consistency_wo_gemini,
        "category_consistency_wo_gemini": category_to_consistency_wo_gemini,
        "recall_statements_wo_gemini": recall_statements_wo_gemini,
        "consistency_statements_wo_gemini": consistency_statements_wo_gemini,
        "num_statements": len(statements_list),
    }


def eval_vixmo_caption2(
        example_idx: int,
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
    statements_list, category_list = get_canonical_statements(prediction, openai_api_key)

    data = data['data']
    recall, category_to_recall, recall_statements = eval_caption_recall(prediction, data, openai_api_key)
    consistency, category_to_consistency, consistency_statements = eval_caption_consistency(statements_list, category_list, data, openai_api_key)

    return {
        "example_idx": example_idx,
        "recall"     : recall,
        "category_to_recall": category_to_recall,
        "consistency": consistency,
        "category_to_consistency": category_to_consistency,
        "recall_statements": recall_statements,
        "consistency_statements": consistency_statements,
        "num_statements": len(statements_list),
    }


