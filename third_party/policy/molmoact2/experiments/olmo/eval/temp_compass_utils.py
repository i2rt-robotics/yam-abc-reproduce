# From https://github.com/llyx97/TempCompass

from typing import Dict, Any, Union
from olmo.eval.api_utils import get_chat_response


caption_evaluation_prompt = """
You will receive a video description and a multi-choice question. Your task is to choose the correct answer and briefly explain the reason why you choose the answer. \
If none of the choice candidates are correct or the video description lacks enough information to answer the question, just answer "None of the choices are correct". \
Please organize your response in this format:
```
Reasoning: [Your reason to obtain the answer]
Answer: [Your answer]
```

Here are some examples of video description, multi-choice question and the expected answer:
```
Video Description: A person is palying football.
Multi-Choice Question:
What is the person doing in the video?
A. cooking
B. palying football
C. playing basketball
D. reading book
Reasoning: The video description mentions that the person is playing football.
Answer: B. palying football

Video Description: A bird is flying clockwise.
Multi-Choice Question:
In which direction is the bird flying?
A. backwark
B. counter-clockwise
C. clockwise
D. downward
Reasoning: The video description mentions that the bird is flying clockwise
Answer: C. clockwise

Video Description: An air balloon is inflating.
Multi-Choice Question:
What is happening to the air balloon?
A. exploding
B. getting smaller
C. flying
Reasoning: The video description mentions that the air balloon is inflating, while none of the coices can be explained as inflating.
Answer: None of the choices are correct
```
"""


multi_choice_evaluation_prompt = """
You will receive a multi-choice question, the ground-truth answer and the prediction from a question answering (QA) model. \
Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
"""


yes_no_evaluation_prompt = """
You will receive a Yes/No question, the ground-truth answer and the prediction from a question answering (QA) model. \
Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
"""


caption_matching_evaluation_prompt = """
You will receive a caption matching question, the ground-truth answer and the prediction from a question answering (QA) model. \
Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
"""


# ----------- Non-captioning task gpt evaluation -------------
def llm_output_to_rating(llm_output: str) -> int:
    if 'Correct' not in llm_output and 'Incorrect' not in llm_output:
        rating = 0
    elif llm_output.startswith('Correct'):
        rating = 1
    elif llm_output.startswith('Incorrect'):
        rating = 0
    elif ('Correct' in llm_output) and ('Incorrect' not in llm_output):
        rating = 1
    elif 'Incorrect' in llm_output:
        rating = 0
    return rating


def get_llm_output(
    prompt: str, openai_api_key: str, system_prompt: str=None, max_tokens: int=128, maxtry: int=10
) -> str:
    if system_prompt is None:
        system_prompt = "You are an AI assistant for question answering."
    gen_params = dict(
        model="gpt-3.5-turbo-1106",
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1,
        presence_penalty=1,
        patience=maxtry,
    )
    llm_output = get_chat_response(prompt, openai_api_key, system_prompt=system_prompt, **gen_params)
    return llm_output


def get_eval_result(prompt: str, openai_api_key: str, maxtry: int=10, system_prompt: str=None) -> int:
    llm_output = get_llm_output(prompt, openai_api_key, system_prompt, maxtry=maxtry)
    rating = llm_output_to_rating(llm_output)
    return rating


# ----------- Captioning task gpt evaluation -------------
def parse_llm_output(llm_output: str, gt_answer: str) -> Dict[str, Any]:
    if not llm_output:
        eval_result = {"rating": -1, "chatgpt-answer": None, "chatgpt-reasoning": None}
        return eval_result

    eval_result = {}
    lines = llm_output.split("\n")

    for line in lines:
        line = line.strip()
        if "Reasoning" in line:
            eval_result["chatgpt-reasoning"] = line.replace("Reasoning:", "").strip()
        if "Answer" in line:
            eval_result["chatgpt-answer"] = line.replace("Answer:", "").strip()

    if not "chatgpt-answer" in eval_result:
        eval_result["chatgpt-answer"] = llm_output
    if not "chatgpt-reasoning" in eval_result:
        eval_result["chatgpt-reasoning"] = None

    # Check if the chatgpt answer is the ground-truth answer
    answer_counts = sum(eval_result["chatgpt-answer"].count(prefix) for prefix in ['A.', 'B.', 'C.', 'D.']) # calculate the number of 'A.', 'B.', 'C.', 'D.' in chatgpt-answer
    if eval_result["chatgpt-answer"].split(". ")[0]==gt_answer.split(". ")[0] and answer_counts==1:
        eval_result["rating"] = 1
    else:
        eval_result["rating"] = 0
    return eval_result


def get_captioning_eval_result(prompt: str, mc_answer: str, openai_api_key: str, maxtry: int=10) -> int:
    llm_output = get_llm_output(
        prompt, openai_api_key, maxtry=maxtry,
    )
    eval_result = parse_llm_output(llm_output, mc_answer)
    return eval_result["rating"]


# ----------- Task evaluation -------------
def eval_multi_choice(
    question: str,
    prediction: str,
    answer: str,
    openai_api_key: str,
    use_api: bool = True,   
) -> float:
    if prediction == answer:
        rating = 1
    elif prediction in ["A", "B", "C", "D"]:
        rating = 1 if prediction == answer[0] else 0
    elif any(prediction.startswith(prefix) for prefix in ['A.', 'B.', 'C.', 'D.']):
        rating = 1 if prediction.split('.')[0] == answer[0] else 0
    elif any(prediction.startswith(prefix) for prefix in ['A)', 'B)', 'C)', 'D)']):
        rating = 1 if prediction.split(')')[0] == answer[0] else 0
    elif not use_api:  # Fail to match answer in the video-llm response. Directly set rating to 0
        rating = 0
    else:  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        prompt = f"""{multi_choice_evaluation_prompt}\nMulti-Choice Question:\n{question}\nGround-Truth Answer: {answer}\nModel Prediction: {prediction}"""
        rating = get_eval_result(prompt, openai_api_key)
    
    return float(rating)


def extract_pred(prediction: str) -> Union[str, bool]:
    """Extract the yes/no predction from the original video llm output"""
    pred = prediction.lower()
    if pred.startswith("yes"):
        return "yes"
    elif pred.startswith("no"):
        return "no"
    else:
        False


def eval_yes_no(
    question: str,
    prediction: str,
    answer: str,
    openai_api_key: str,
    use_api: bool = True,          
) -> float:
    yes_no_pred: Union[str, bool] = extract_pred(prediction)  # Some hand-crafted matching rules
    if yes_no_pred:
        rating = 1 if yes_no_pred == answer else 0
    elif not use_api:  # Fail to match answer in the video-llm response. Directly set rating to 0
        rating = 0
    else:  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        prompt = f"""{yes_no_evaluation_prompt}\nYes/No Question:\n{question}\nGround-Truth Answer: {answer}\nModel Prediction: {prediction}"""
        rating = get_eval_result(prompt, openai_api_key)
    return float(rating)


def parse_caption_matching_output(prediction: str, question: str) -> str:
    """Parse caption matching put based on word matching rules"""
    option_strs = question.split("\n")[1:]  # complete option strings
    option_sents = [opt.split(': ')[1] for opt in option_strs]  # option sentence
    option_inds = [opt.split(': ')[0] for opt in option_strs]
    option_inds += [
        opt.split(': ')[0]
        .replace('Sentence ', '')
        .replace('Option ', '')
        .replace('Caption ', '')
        for opt in option_strs
    ]   # option index, e.g., Sentence A, Caption A, Option 1
    parsed = None
    for option_str in option_strs:
        if option_str == prediction:
            parsed = option_str
    for option_sent in option_sents:
        if option_sent == prediction or (') ' in prediction and option_sent == prediction.split(') ')[1]):
            parsed = option_sent
    for option_ind in option_inds:
        if option_ind == prediction or option_ind == prediction.replace('.', ''):
            parsed = option_ind
    return parsed


def eval_caption_matching(
    question: str,
    prediction: str,
    answer: str,
    openai_api_key: str,
    use_api: bool = True,   
) -> float:
    parsed = parse_caption_matching_output(prediction, question)
    if parsed is not None:
        rating = int(
            parsed == answer or 
            parsed == answer.split(":")[0] or 
            parsed == answer.split(": ")[1] or
            parsed == answer.split(": ")[0].split()[1]
        )
    elif not use_api:  # Fail to match answer in the video-llm response. Directly set rating to 0
        rating = 0
    else:  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        prompt = f"""{caption_matching_evaluation_prompt}\nCaption Matching Question:\n{question}\nGround-Truth Answer: {answer}\nModel Prediction: {prediction}"""
        rating = get_eval_result(prompt, openai_api_key)
    return float(rating)


def eval_captioning(
    prediction: str,
    mc_question: str,
    mc_answer: str,
    openai_api_key: str,   
) -> float:
    prompt = f"""{caption_evaluation_prompt}\nVideo Description:{prediction}\nMulti-Choice Question:\n{mc_question}\n"""
    rating = get_captioning_eval_result(prompt, mc_answer, openai_api_key)
    return float(rating)


def temp_compass_score(
    prediction: str,
    metadata: dict,
    openai_api_key: str,
    use_api: bool = True,
) -> float:
    task = metadata["task"]
    question = metadata["question"]
    answer = metadata["answer"]
    mc_question = metadata.get("mc_question", None)
    mc_answer = metadata.get("mc_answer", None)

    if task in ["multi-choice", "caption_matching"]:
        score = eval_multi_choice(
            question, prediction, answer, openai_api_key, use_api
        )
    elif task == "yes_no":
        score = eval_yes_no(
            question, prediction, answer, openai_api_key, use_api
        )
    else:
        assert mc_question is not None and mc_answer is not None
        score = eval_captioning(
            prediction, mc_question, mc_answer, openai_api_key
        )
    return score
