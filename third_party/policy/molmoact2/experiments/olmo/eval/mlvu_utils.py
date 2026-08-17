# From https://github.com/JUNJIE99/MLVU

from typing import Dict, Any, Union, List
from olmo.eval.api_utils import get_chat_response


sub_scene_system_prompt = """
    ##TASK DESCRIPTION: 
    You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.
    ##ACCURACY Scoring Criteria: 
    Evaluate the respondent's answer against specific scoring points as follows:
    Score 1: The response completely misses the scoring point.
    Score 3: The response mentions content related to the scoring point but is not entirely correct.
    Score 5: The response accurately addresses the scoring point.
    Calculate the average score across all scoring points to determine the final accuracy score.
    ##RELEVANCE Scoring Criteria:
    Assess how the respondent's answer relates to the original question:
    Score 1: The response is completely off-topic from the question.
    Score 2: The response is partially related to the question but contains a significant amount of irrelevant content.
    Score 3: The response primarily addresses the question, but the respondent seems uncertain about their own answer.
    Score 4: The response mostly addresses the question and the respondent appears confident in their answer.
    Score 5: The response is fully focused on addressing the question with no irrelevant content and demonstrates complete certainty.
    ----
    ##INSTRUCTION:
    1. Evaluate Accuracy: First, assess and score each scoring point based on the respondent's answer. Calculate the average of these scores to establish the final accuracy score. Provide a detailed rationale before assigning your score.
    2. Evaluate RELEVANCE: Assess the relevance of the respondent's answer to the question. Note that when evaluating relevance, the correctness of the answer is not considered; focus solely on how relevant the answer is to the question. Provide a comprehensive rationale before assigning your score.
    3. Output Scores in JSON Format: Present the scores in JSON format as follows:
    {'score_accuracy': score_acc, 'score_relevance': score_rele, 'total_score': score_acc + score_rele}
"""


summary_system_prompt = """
    ##TASK DESCRIPTION: 
    You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
    ##COMPLETENESS Scoring Criteria:
    The completeness score focuses on whether the summary covers all key points and main information from the video. 
    Score 1: The summary hardly covers any of the main content or key points of the video.
    Score 2: The summary covers some of the main content and key points but misses many.
    Score 3: The summary covers most of the main content and key points.
    Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
    Score 5: The summary completely covers all the main content and key points of the video.
    ##RELIABILITY Scoring Criteria:
    The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
    Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
    Score 2: Includes several errors and some contradictions; needs clearer presentation.
    Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
    Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
    Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
    ----
    ##INSTRUCTION:
    1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
    2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
    3. Output Scores in JSON Format: Present the scores in JSON format as follows:
    {'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
"""


def get_llm_output(prompt: str, openai_api_key: str, system_prompt: str) -> str:
    """
    Get the response from the LLM model.

    Args:
    prompt (str): The prompt to send to the model.
    openai_api_key (str): OpenAI API key.
    system_prompt (str): The system prompt to use.

    Returns:
    str: The response from the model.
    """

    response = get_chat_response(
        model="gpt-4-turbo",
        temperature=0,
        prompt=prompt,
        api_key=openai_api_key,
        system_prompt=system_prompt,
    )
    return response


def parse_response(text: str, keys: List[str]) -> List[float]:
    scores = []

    for key in keys:
        # Find the index where each key starts
        start_index = text.find(key)
        if start_index == -1:
            scores.append(0)  # Append 0 if the key is not found
            continue

        # Find the start of the number which is after the colon and space
        start_number_index = text.find(":", start_index) + 2
        end_number_index = text.find(",", start_number_index)  # Assuming the number ends before a comma

        # Extract and convert the number to float
        try:
            score = float(text[start_number_index:end_number_index])
        except:
            score = 0
        scores.append(score)
    return scores


def mlvu_ssc_score(
    prediction: str,
    metadata: dict,
    openai_api_key: str,
) -> dict:
    """
    Calculate the accuracy and relevance scores for the SSC task in MLVU.

    Args:
    prediction (str): The response to evaluate.
    metadata (dict): Metadata containing the scoring points and question.
    openai_api_key (str): OpenAI API key.

    Returns:
    dict: The accuracy and relevance scores.
    """
    question = metadata["question"].replace("\n", "")
    scoring_points = metadata["scoring_points"]
    prompt = f"""
        Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
        Question: {question}
        Scoring Points: {scoring_points}
        Respondent's Answer: {prediction}
    """
    response = get_llm_output(prompt, openai_api_key, sub_scene_system_prompt)

    # Define the keys to locate in the text
    keys = ["score_accuracy", "score_relevance"]
    scores = parse_response(response, keys)

    acc = scores[0]
    rel = scores[1]

    return dict(sub_scene_accuracy=acc, sub_scene_relevance=rel)


def mlvu_summary_score(
    prediction: str,
    metadata: dict,
    openai_api_key: str,
) -> dict:
    """
    Calculate the completeness and reliability scores for the Summary task in MLVU."

    Args:
    prediction (str): The response to evaluate.
    metadata (dict): Metadata containing the answer.
    openai_api_key (str): OpenAI API key.

    Returns:
    dict: The completeness and reliability scores.
    """
    answer = metadata["answer"]

    prompt = f"""
        Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
        Standard Answer: {answer}
        Respondent's Answer: {prediction}
    """

    response = get_llm_output(prompt, openai_api_key, summary_system_prompt)

    # Define the keys to locate in the text
    keys = ["score_completeness", "score_reliability"]
    scores = parse_response(response, keys)

    comp = 0
    rel = 0
    try:
        comp += scores[0]
        rel += scores[1]
    except:
        pass
    return dict(summary_completeness=comp, summary_reliability=rel)