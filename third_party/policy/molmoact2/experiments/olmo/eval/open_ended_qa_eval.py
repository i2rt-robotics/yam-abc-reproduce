import argparse
import base64
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os.path import join, exists, basename, relpath

from tqdm import tqdm

from olmo.config import StrEnum
from olmo.eval.evaluators import Evaluator, mean_metric, gather_examples_as_html
from olmo.html_utils import build_html_table
from olmo.io import file_exists, read_file, write_file
from olmo.torch_util import get_world_size, get_global_rank
from olmo.util import compute_hash, flatten_list, resource_path
import torch.distributed as dist


log = logging.getLogger(__name__)


class GptWithCache:
    def __init__(self, model, cache_dir=None, cache_only=False):
        if cache_dir is None:
            cache_dir = join(os.environ["MOLMO_DATA_DIR"], "gpt_cache")
        self.model = model
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        import openai  # import here so dependency is optional
        self.client = openai.OpenAI()

    def __call__(self, message, **kwargs):
        if isinstance(message, str) and len(kwargs) == 0:
            query_hash = compute_hash(self.model + "::::" + message)
        else:
            query_hash = compute_hash(self.model + "::::" + json.dumps(message) + "::::" + json.dumps(kwargs, sort_keys=True))
        use_cache = self.cache_dir

        if use_cache:
            cache_file = join(self.cache_dir, f"{query_hash}-v1.json")
            if file_exists(cache_file):
                return json.loads(read_file(cache_file))

        if self.cache_only:
            raise ValueError("Not cached")

        if isinstance(message, str):
            message = [{"role": "user", "content": message}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            **kwargs
        )
        completion = completion.model_dump()

        if use_cache:
            write_file(self.cache_dir, basename(cache_file), json.dumps(completion), True)
        return completion


class ImageEvalOptions(StrEnum):
    correct = "correct"
    mostly_correct = "mostly correct"
    mostly_incorrect = "mostly incorrect"
    incorrect = "incorrect"
    unable_to_answer = "refuse to answer"


def eval_question_against_image(_args):
    ex_id, _, gpt, question, image, answer = _args
    ext = image.split(".")[-1]
    with open(resource_path(image), "rb") as image_file:
        image_bytes = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = f"""
Question: {question}
Answer: {answer}

Look at the image carefully, then evaluate whether the answer to the question about this image is correct. 
- Focus on whether the answer accurately reflects the content of the image. 
- Focus on factual errors, do not penalize the model for answers with interpretations or subjective statements that are debatable, but not clearly incorrect.
- If the answers states it is unable to answer the question, consider that correct if it really would be impossible to answer the question with the image alone.

Answer by giving a very brief explanation, it should usually be one sentence, then a new line, and then one of these options: {", ".join(ImageEvalOptions)}

If you have to refuse to answer, use the \"refuse to answer\" option.

Do not output any other text.  
""".strip()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{image_bytes}",
                        "detail": "high"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = gpt(messages)
    parts = response["choices"][0]["message"]["content"].strip()
    explanation, evaluation = parts.rsplit("\n", 1)
    evaluation = evaluation.lower().strip()
    for opt in ImageEvalOptions:
        if opt.lower() == evaluation:
            return opt, explanation
    return None, parts


class GtEvalOptions(StrEnum):
    correct = "correct"
    mostly_correct = "mostly correct"
    mostly_incorrect = "mostly incorrect"
    incorrect = "incorrect"
    unsure = "unsure"
    unable_to_answer = "refuse to answer"


def eval_question_against_gt(_args):
    ex_id, _, gpt, question, gt_answer, answer = _args
    prompt = f"""
Question: {question}
Answer: {answer}
Ground truth: {gt_answer}

Decide if the ground truth answer is consistent with the answer.
- Focus on the main point of the answers. It is okay if the ground truth answer contains details not in the answer, and vice versa, as long as they give the same overall response.
- If the answers are different but do not contradict one another, respond with unsure. 

Answer by giving a very brief explanation, it should usually be one sentence, then a new line, and then one of these options: {", ".join(GtEvalOptions)}

If you have to refuse to answer, use the \"refuse to answer\" option.

Do not output any other text.  
""".strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = gpt(messages)
    parts = response["choices"][0]["message"]["content"].strip()
    explanation, evaluation = parts.rsplit("\n", 1)
    evaluation = evaluation.lower().strip()
    for opt in ImageEvalOptions:
        if opt.lower() == evaluation:
            return opt, explanation
    return None, parts


def eval_question(_args):
    ex_id, is_image = _args[:2]
    if is_image:
        score, resp = eval_question_against_image(_args)
    else:
        score, resp = eval_question_against_gt(_args)
    return ex_id, is_image, score, resp


class OpenQaEvaluator(Evaluator):
    def __init__(self, n_to_log=None, total_threads=128, model="gpt-5", output_dir=None,
                 pbar=False, skip_how_many=True):
        self.n_to_log = n_to_log
        self.n_threads = max(1, total_threads // get_world_size())
        self.output_dir = output_dir
        self.pbar = pbar
        self.skip_how_many = skip_how_many
        self.gpt = GptWithCache(model)

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        vocab = tokenizer

        if self.skip_how_many:
            keep = []
            for i, metadata in enumerate(metadatas):
                if not metadata["question"].strip().lower().startswith("how many"):
                    keep.append(i)
            metadatas = [metadatas[i] for i in keep]
            new_tokens = [new_tokens[i] for i in keep]

        _args = []
        responses = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            metadata = metadatas[ex_ix]
            _args.append((ex_ix, True, self.gpt, metadata["question"],
                          metadata["image_file"], pred))
            _args.append((ex_ix, False, self.gpt, metadata["question"],
                          metadata["answer"], pred))
            responses.append(dict(answer=pred, metadata=metadata))

        scores = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            for ex_ix, is_image, score, explanation in tqdm(
                pool.map(eval_question, _args),
                total=len(_args), disable=not self.pbar
            ):
                scores["valid"].append(score is not None)
                metadata = metadatas[ex_ix]
                if is_image:
                    responses[ex_ix].update(
                        image_explanation=explanation,
                        image_score=score,
                    )
                    name = "img"
                else:
                    responses[ex_ix].update(
                        text_explanation=explanation,
                        text_score=score,
                    )
                    name = "txt"
                if score is not None:
                    scores[f"{name}_overall_correct"].append(
                        score in [ImageEvalOptions.mostly_correct, ImageEvalOptions.correct])
                    scores[f"{name}_overall_incorrect"].append(
                        score in [ImageEvalOptions.mostly_incorrect, ImageEvalOptions.incorrect])
                    for opt in ImageEvalOptions:
                        scores[f"{name}_{opt}"].append(score == opt)

        if self.output_dir:
            log.info("Save all evaluations")
            if get_world_size() > 1:
                if get_global_rank() == 0:
                    global_responses = [None]*get_world_size()
                    dist.gather_object(responses, global_responses)
                    global_responses = flatten_list(global_responses)
                else:
                    dist.gather_object(responses, None)
                    global_responses = None
            else:
                global_responses = responses

            if get_global_rank() == 0:
                eval_file = join(self.output_dir, "full_eval.json")
                log.info(f"Saving json responses to {eval_file}")
                write_file(
                    eval_file,None,
                    json.dumps(global_responses, indent=2),
                    save_overwrite=True
                )

        for ex in responses:
            image_score = ex["image_score"]
            gt_score = ex["text_score"]
            gt_score_c = gt_score in [GtEvalOptions.correct, GtEvalOptions.mostly_correct]
            img_score_c = image_score in [ImageEvalOptions.correct, ImageEvalOptions.mostly_correct]
            if image_score == ImageEvalOptions.unable_to_answer:
                overall_score = gt_score_c
            elif gt_score in [GtEvalOptions.unable_to_answer, GtEvalOptions.unsure]:
                overall_score = img_score_c
            else:
                overall_score = img_score_c or gt_score_c
            scores["overall_score"].append(overall_score)

        out = {k: mean_metric(v) for k, v in scores.items()}
        return out
