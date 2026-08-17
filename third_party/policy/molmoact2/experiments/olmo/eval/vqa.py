"""Various VQA metrics used by different datasets"""
import re
import string
from collections import Counter
from typing import Optional, List, Tuple, Dict
import random
import editdistance
import numpy as np

from olmo.eval import mmmu_eval_utils, math_vista_utils

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
    "youll": "you'll", "youre": "you're", "youve": "you've"}

manualMap = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a','an','the']

punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!']

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",outText,re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(ans, cache={}):
    if ans in cache:
        return cache[ans]
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t',' ')
    ans = ans.lower().strip()
    preprocessed = processDigitArticle(processPunctuation(ans))
    cache[ans] = preprocessed
    return preprocessed


def vqa_score(target, pred):
    """
    Evaluation with VQA 2 style preprocessing
    """
    pred = preprocess_answer(pred)
    if isinstance(target, list):
        target = Counter(preprocess_answer(x) for x in target)
        return min(target[pred] / 3.0, 1)
    else:
        return float(pred == target)


def a_okvqa_score(target, pred):
    # A-OK-VQA eval scripts don't seem to do any answer pre-processing
    target = Counter([x.lower().strip() for x in target])
    return min(target[pred.lower().strip()] / 3.0, 1)


def select_mc_option(target, options):
    """
    Selects a multiple-choice option based on the model output

    The output is should exactly match one of the option, but contains
    some heuristic fallbacks in case the does not occur
    """
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    assert len(set(options)) == n
    for ix, option in enumerate(options):
        if option == target:
            return ix

    contains = []
    for ix, option in enumerate(options):
        if target.startswith(option):
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]

    contains = []
    for ix, option in enumerate(options):
        if option.startswith(target):
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]

    contains = []
    for ix, option in enumerate(options):
        if target in option:
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]
    distances = [editdistance.eval(opt, target) for opt in options]
    return np.argmin(distances)


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """
    # Lowercase is not in https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
    # However https://rrc.cvc.uab.es/?ch=17&com=tasks says
    #  - "Answers are not case sensitive"
    #  - "Answers are space sensitive"
    edit_distance = editdistance.eval(target.lower(), prediction.lower())
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def scifi_relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05) -> bool:

    def _to_float(text: str) -> Optional[float]:
        try:
            return float(text)
        except ValueError:
            return None

    def compute_relative_change(target: float, prediction: float) -> float:
        if target == 0: return abs(target - prediction)
        return abs(target - prediction) / abs(target)
    
    def extract_short_answer(prediction: float) -> float:
        if "answer:" in prediction: return prediction.split("answer:")[1].strip()
        else: return prediction

    prediction = extract_short_answer(prediction.lower().strip())
    target = extract_short_answer(target.lower().strip())

    if len(prediction) == 0:
        return False

    if prediction[-1] == ".":
        prediction = prediction[:-1]
    
    WORD_TO_NUM = {k: v for k, v in manualMap.items() if k != "none"}

    target_float = _to_float(target)
    if target_float is not None:
        # target is a float number
        if "," in prediction: prediction = prediction.replace(",", "") # remove commas

        # map words to numbers
        for word, num in WORD_TO_NUM.items():
            prediction = prediction.replace(word, str(num))

        # extract the number from the prediction, considering float numbers (.), using regex
        try: prediction_float = _to_float(re.search(r"[-+]?\d*\.\d+|\d+", prediction).group())
        except: return False # if no number is found, return False

        relative_change = compute_relative_change(target_float, prediction_float)

        prediction_float_normalized = prediction_float / 100
        relative_change_normalized = compute_relative_change(target_float, prediction_float_normalized)

        if relative_change <= max_relative_change or relative_change_normalized <= max_relative_change:
            return True
        else:
            return False

    else:
        # target is a string
        if "[" in target and "," in target:
            # target is a list
            target = target.replace("[", "").replace("]", "")
            targets = target.split(",")
            correct = True

            for t in targets:
                if t.strip().lower() not in prediction:
                    correct = False
                    break

            if correct:
                return True
            else:
                return False

        else:
            if target.strip().lower() in prediction:
                return True
            else:
                return False


# From https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/main_parse_and_eval.py
def mmmu_score(
    target: List[str],
    response: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple-choice":
        options = metadata["options"]
        options = [opt for opt in options if len(opt) > 0]
        all_choices = [chr for chr in string.ascii_uppercase[:len(options)]]
        index2ans = {chr: option for chr, option in zip(all_choices, options)}
        parsed_pred = mmmu_eval_utils.parse_multi_choice_response(response, all_choices, index2ans)
        correct = mmmu_eval_utils.eval_multi_choice(target, parsed_pred)
    else: # open
        parsed_pred = mmmu_eval_utils.parse_open_response(response)
        correct = mmmu_eval_utils.eval_open(target, parsed_pred)
    return float(correct)


def real_world_qa_score(
    target: str,
    prediction: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple_choice":
        options = ["A", "B", "C", "D"]
        pred_idx = select_mc_option(prediction, options)
        gt_idx = options.index(target)
        score = pred_idx == gt_idx
    else:
        pred = preprocess_answer(prediction)
        gt = preprocess_answer(target)
        score = float(pred == gt)
    return score


def math_vista_score(
    response: str,
    metadata: dict,
    openai_api_key: str,
    use_api: bool = True,
):
    # extract answer using GPT-4.
    pid = metadata["example_id"]
    question_type = metadata["question_type"]
    answer_type = metadata["answer_type"]
    choices = metadata["choices"]
    target = metadata["answer"]
    query = metadata["query"]

    if use_api:
        extraction = math_vista_utils.extract_answer(
            pid, response, question_type, answer_type, choices, query, openai_api_key,
        )
    else:
        if question_type == "multi_choice":
            options = [chr(ord("A") + i) for i in range(len(choices))]
            pred_idx = select_mc_option(response, options)
            extraction = choices[pred_idx]
        else:
            if answer_type == "integer":
                try:
                    extraction = str(int(response))
                except:
                    extraction = response
            elif answer_type == "float":
                try:
                    extraction = str(float(response))
                except:
                    extraction = response
            else:
                extraction = response

    # calculate score
    precision = metadata["precision"]

    # normalize the extracted answer to match the answer type
    prediction = math_vista_utils.normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision,
    )

    # verify the prediction is true or false
    true_false = math_vista_utils.safe_equal(prediction, target)

    return true_false


def mlvu_mc(target: str, prediction: str) -> bool:
    
    def _extract_characters_regex(s: str) -> str:
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s
    
    pred_ans = _extract_characters_regex(prediction)
    return target == pred_ans


def select_perception_test_option(prediction: str) -> int:
    pred = prediction.strip() # raw text prediction

    # Use regex to match A, B, C, or D
    match = re.search(r"\b([A-D])\b", pred)

    if match:
        pred = match.group(1)  # Extract the matched letter
        pred = pred.upper()
    else:
        pred = ""  # Set to empty string if no match found

    # Map the prediction to an index
    pred_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    index = pred_to_index.get(pred, -1)  # Default to -1 if the prediction is not found
    return index


def ego_schema_get_multi_choice_info(options: List[str]) -> Tuple[Dict[str, str], List[str]]:
    all_choices = []
    index2ans = {}
    OPTIONS = ["A", "B", "C", "D", "E"]
    for i in range(5):
        index2ans[OPTIONS[i]] = options[i].strip()
        all_choices.append(OPTIONS[i])
    
    return index2ans, all_choices


def ego_schema_parse_multi_choice_response(
    response: str,
    all_choices: List[str],
    index2ans: Dict[str, str],
) -> Tuple[str, bool]:
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_space = False
    ans_with_dot = False
    candidates = []

    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(f"({choice})")
            ans_with_brack = True

    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            candidates.append(f"{choice} ")
            ans_with_space = True

    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(f"{choice}.")
            ans_with_dot = True

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for can in candidates:
                index = response.rfind(can)
                start_indexes.append(index)  # -1 will be ignored anyway
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the first one
        pred_index = candidates[np.argmin(start_indexes)]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()
    else:  # if only one candidate, use it.
        pred_index = candidates[0]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()
    
    return pred_index, len(candidates) > 0


def select_ego_schema_option(prediction: str, options: List[str]) -> int:
    index2ans, all_choices = ego_schema_get_multi_choice_info(options)
    parsed_pred, matched_tag = ego_schema_parse_multi_choice_response(prediction, all_choices, index2ans)
    
    pred_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    index = pred_to_index.get(parsed_pred, -1)  # Default to -1 if the prediction is not found
    return index


def nextqa_get_multi_choice_info(options: List[str]) -> Tuple[Dict[str, str], List[str]]:
    all_choices = []
    index2ans = {}
    OPTIONS = ["A", "B", "C", "D", "E"]
    for i in range(len(OPTIONS)):
        index2ans[OPTIONS[i]] = options[i].strip()
        all_choices.append(OPTIONS[i])
    
    return index2ans, all_choices


def nextqa_parse_multi_choice_response(response: str, all_choices: List[str], index2ans: Dict[str, str]) -> str:
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_space = False
    ans_with_dot = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)
                ans_with_space = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)
                ans_with_dot = True

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            elif ans_with_space:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def nextqa_mc(target: str, prediction: str, options: List[str]) -> bool:
    index2ans, all_choices = nextqa_get_multi_choice_info(options)
    parsed_pred = nextqa_parse_multi_choice_response(prediction, all_choices, index2ans)
    return target == parsed_pred


def muir_bench_get_multi_choice_info(options: List[str]) -> Tuple[Dict[str, str], List[str]]:
    all_choices = []
    index2ans = {}
    OPTIONS = list(string.ascii_uppercase)[:len(options)]
    # Removed since MMIU uses this same method and it can have more options
    # assert len(options) <= 5, "MuirBench only supports 5 options"
    for i in range(len(options)):
        index2ans[OPTIONS[i]] = options[i].strip()
        all_choices.append(OPTIONS[i])
    
    return index2ans, all_choices


def muir_bench_parse_multi_choice_response(
    response: str,
    all_choices: List[str],
    index2ans: Dict[str, str],
    fallback_to_random: bool = True,
) -> str:
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_last_brack = False
    ans_with_dot = False
    ans_with_colon = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    for choice in all_choices: # e.g., A), B), C), D)
        if f"{choice})" in response:
            candidates.append(choice)
            ans_with_last_brack = True
    
    for choice in all_choices: # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_dot = True

    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if response.strip() == choice:
                return choice
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0 and fallback_to_random:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) == 0:
        pred_index = 'N/A'
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
            elif ans_with_last_brack:
                for can in candidates:
                    index = response.rfind(f"{can})")
                    start_indexes.append(index)
            elif ans_with_dot:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def muir_bench_mc(
    target: str,
    prediction: str,
    options: List[str],
    fallback_to_random: bool = True,
) -> bool:
    index2ans, all_choices = muir_bench_get_multi_choice_info(options)
    parsed_pred = muir_bench_parse_multi_choice_response(
        prediction, all_choices, index2ans, fallback_to_random
    )
    return target == parsed_pred


def mantis_eval_mc(target: str, prediction: str, options: List[str], question_type: str) -> bool:
    if question_type == "multi-choice":
        return muir_bench_mc(target, prediction, options)
    else:
        pred = preprocess_answer(prediction)
        gt = preprocess_answer(target)
        return float(pred == gt)


def ego3d_bench_extract_number(text: str) -> int | float:
    # Extract the first number (int or float) from the content
    number_match = re.search(r'[-+]?\d*\.?\d+', text)
    if number_match:
        return float(number_match.group()) if '.' in number_match.group() else int(number_match.group())
    return None


def ego3d_bench_eval_exact_num(target: str, prediction: str) -> float:
    num_pred = ego3d_bench_extract_number(prediction)
    if num_pred:
        num_pred = min(float(num_pred), 100)
        gt = float(target)
        return (gt - num_pred) ** 2
    return -1


def ego3d_bench_score(target: str, prediction: str, task: str, options: List[str] | None) -> bool:
    if task == "number":
        score = ego3d_bench_eval_exact_num(target, prediction)
    else:
        score = muir_bench_mc(target, prediction, options)
    return score