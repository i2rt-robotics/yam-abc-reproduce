import json
import logging
import time
from typing import Dict

import openai

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
                prediction = response.choices[0].message.content
                if prediction != "" and prediction != None:
                    return json.loads(prediction)
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


def call_azure_gpt_api(events, reference, prediction, openai_api_key, maxtry: int = 100):
    if len(events) == 0:
        events = [reference.replace('\n', ' ')]

    prompt = (
        "Given a video description and a list of events. For each event, classify the relationship between the video description and the event into three classes: entailment, neutral, contradiction.\n"
        "- \"entailment\" means that the video description entails the event.\n"
        "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
        "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
        f"Video Description:\n{prediction}\n\n"
        f"Events: {events}\n"

        "Output a JSON formed as:\n"
        "{\n"
        "  \"events\": [\n"
        "    {\"event\": \"copy an event here\", \"relationship\": \"put class name here\",  \"reason\": \"give your reason here\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
    )
    gen_params = dict(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        patience=maxtry,
        response_format={"type": "json_object"}

    )
    llm_output = get_chat_response(prompt, openai_api_key, system_prompt=None, **gen_params)
    return llm_output


def evaluate_one_sample(events, response, prediction, openai_api_key, return_hit_num=False, is_recall=False, max_retry=10):
    events_filled = call_azure_gpt_api(events, response, prediction, openai_api_key)
    events_filled = events_filled['events']
    num_matched_events = 0
    try:
        for event in events_filled:
            pred = event['relationship'].strip().lower()
            assert pred in ['entailment', 'neutral', 'contradiction']
            pos_classes = ['entailment'] if is_recall else ['entailment', 'neutral']
            if pred in pos_classes:
                num_matched_events += 1
    except Exception as e:
        log.warning(f"Invalid response: {events_filled}")
    if len(events) == 0:
        motion_score = 1.0
    else:
        motion_score = num_matched_events / len(events)
    if return_hit_num:
        return motion_score, events_filled, f"hit: {num_matched_events} / {len(events)}"
    return motion_score


def call_azure_gpt_api_for_events(caption, openai_api_key, maxtry: int = 100):
    prompt = ("Bellow is a description of a video clip:\n"
              f"Video Description: {caption}\n\n"

              "Extract at most 10 key events from the above video description paragraph. Requirements\n:"
              "- An event must include an action, motion or movement (NOT STATIC INFORMATION). DON'T repeat same events.\n"
              "- Every event is represented by a brief sentence within 10 words, with a subject, a predicate and optionally an object, avoid unnecessary appearance descriptions.\n"
              "- Every event must be atomic, meaning that it cannot be further split into multiple events.\n"
              "- Scene cuts and camera motions are NOT events.\n"
              "- Substitute pronouns by the nouns they refer to.\n\n"
              "Please generate the response in the form of a JSON object with keys \"events\". The value of \"events\" is a List(str), of which each item is an event. "
              "For example, your response should look like this: {\"events\": [event1, event2, ...]}")

    gen_params = dict(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        patience=maxtry,
        response_format={"type": "json_object"}

    )
    llm_output = get_chat_response(prompt, openai_api_key, system_prompt=None, **gen_params)
    return llm_output


def extract_events(caption, openai_api_key):
    caption = caption.replace("\"", "\'")
    result = call_azure_gpt_api_for_events(caption, openai_api_key)
    return result['events']


def eval_dream_caption(
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

    pred_events = extract_events(prediction, openai_api_key)
    gt_events = data['events']
    response = data['description']

    motion_score_r, events_filled_r, hit_num_r = evaluate_one_sample(gt_events, response, prediction, openai_api_key, return_hit_num=True, is_recall=True)
    motion_score_p, events_filled_p, hit_num_p = evaluate_one_sample(pred_events, prediction, response, openai_api_key, return_hit_num=True, is_recall=True)

    return {
        "example_idx"      : example_idx,
        "gt"               : response,
        'pred'             : prediction,
        'events_gt'        : events_filled_r,
        'hit_num_recall'   : hit_num_r,
        'events_pred'      : events_filled_p,
        "hit_num_precision": hit_num_p,
        "score_r"          : motion_score_r,
        "score_p"          : motion_score_p,
    }
