import json
import os
from pathlib import Path
import re
from typing import List, Optional
import string

import pandas as pd
import datasets

_opt_line_re = re.compile(
    r"""^ [^\S\r\n]*
        (?: 
            \( (?P<letter>[A-Ja-j]) \) [^\S\r\n]* [\.:]?      # (A)   or (A). / (A):
          | (?P<letter2>[A-Ja-j]) \) [^\S\r\n]* [\.:]?       # A)    or A). / A):
          | (?P<letter3>[A-Ja-j]) [\.:]                      # A. or A:
        )
        [^\S\r\n]* (?P<text>[^\r\n]*) [^\S\r\n]* $
    """,
    re.MULTILINE | re.VERBOSE,
)
_opt_head_re = re.compile(
    r"""^ [^\S\r\n]*
        (?: 
            \( [A-Ja-j] \) [^\S\r\n]* [\.:]?
          | [A-Ja-j] \)  [^\S\r\n]* [\.:]?
          | [A-Ja-j] [\.:]
        )
    """,
    re.MULTILINE | re.VERBOSE,
)


def keep_question_only(text: str) -> str:
    m = _opt_head_re.search(text)
    return text[:m.start()].rstrip() if m else text


def replace_images(question, max_images=None):
    image_counter = 1
    total_images = question.count("<image>")

    if max_images is not None:
        total_images = min(total_images, max_images)

    def repl(match):
        nonlocal image_counter
        if image_counter > total_images:
            return match.group(0)
        replacement = f"Image {image_counter}"
        image_counter += 1
        return replacement
    return re.sub(r"<image>", repl, question)


def remove_images(question: str):
    return question.replace("<image>", "").strip()


def _pick_letter(m: re.Match) -> str:
    return (m.group('letter') or m.group('letter2') or m.group('letter3'))


def fix_options(questions, questions_no_options, options):
    order = {ch: i for i, ch in enumerate(string.ascii_uppercase[:10])}  # 'A'..'J'

    for i, question in enumerate(questions):
        matches = list(_opt_line_re.finditer(question))
        if len(matches) < 2:
            continue

        questions_no_options[i] = keep_question_only(question)

        picked = {}
        for m in matches:
            L = _pick_letter(m).upper()
            if L in order and L not in picked:
                picked[L] = m.group('text').strip()

        if picked:
            maxL = max(picked, key=lambda c: order[c])
            upto = order[maxL] + 1
            options[i] = [picked.get(string.ascii_uppercase[j], "") for j in range(upto)]

    return questions_no_options, options


class MantisInstructBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="nlvr2", version="1.0.0", description="NLVR2"),
        datasets.BuilderConfig(name="llava_665k_multi", version="1.0.0", description="LLaVA-665k-multi"),
        datasets.BuilderConfig(name="spot-the-diff", version="1.0.0", description="Spot-the-Diff"),
        datasets.BuilderConfig(name="nextqa", version="1.0.0", description="NExT-QA"),
        datasets.BuilderConfig(name="star", version="1.0.0", description="STAR"),
    ]

    TRAIN_ONLY = [
        "llava_665k_multi",
        "spot-the-diff",
        "nextqa",
        "star",
    ]

    YES_OR_NO = [
        "nlvr2",
    ]

    REMOVE_PLACEHOLDER = [
        "llava_665k_multi",
        "nextqa",
        "star",
    ]

    def __init__(self, mantis_instruct_source: str, *args, **kwargs):
        self.mantis_instruct_source = mantis_instruct_source

        super().__init__(*args, **kwargs, dataset_name="mantis_instruct")
    
    def _info(self):
        features = datasets.Features(
            {
                "subset": datasets.Value("string"),
                "example_id": datasets.Value("string"),
                "images": datasets.Sequence(datasets.Value("string")),
                "mc_question": datasets.Sequence(datasets.Value("string")),
                "oe_question": datasets.Sequence(datasets.Value("string")),
                "direct_answer": datasets.Sequence(datasets.Value("string")),
                "choices": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "correct_choice_idx": datasets.Sequence(datasets.Value("int64")),
            }
        )
        return datasets.DatasetInfo(features=features)
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        gen_kwargs = {}
        subset_dir = os.path.join(self.mantis_instruct_source, self.config.name)
        splits = ["train"] if self.config.name in self.TRAIN_ONLY else ["train", "val"]
        for split_name in splits:
            split_gen_kwargs = {}

            json_path = os.path.join(subset_dir, f"{split_name}.json")
            if not Path(json_path).exists():
                json_path = None

            parquet_files = sorted(Path(subset_dir).glob(f"{split_name}*.parquet"))
            split_gen_kwargs["parquet_files"] = parquet_files
            split_gen_kwargs["image_dir"] = os.path.join(subset_dir, f"{split_name}_images")
            split_gen_kwargs["json_path"] = json_path
            split_gen_kwargs["yes_or_no"] = self.config.name in self.YES_OR_NO
            split_gen_kwargs["remove_placeholder"] = self.config.name in self.REMOVE_PLACEHOLDER
            gen_kwargs[split_name] = split_gen_kwargs
        
        outs = [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=gen_kwargs["train"])
        ]
        if "val" in splits:
            outs.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=gen_kwargs["val"]))
        
        return outs
    
    def _generate_examples(
        self, parquet_files: List[str], image_dir: str, json_path: str, yes_or_no: bool, remove_placeholder: bool
    ):
        
        df = pd.read_parquet(parquet_files)
        data = json.load(open(json_path, "r"))

        row_dict = {row["id"]: row for row in df.to_dict(orient="records")}

        for ex in data:
            example_id = ex["example_id"]
            row = row_dict[example_id]
            source = "llava_665k_multi" if row["source"] == "llava_665k_merged" else row["source"]
            if len(set([source, ex["subset"], self.config.name])) != 1:
                raise ValueError(f"Row {example_id} has different source, subset, and config name: {row['source']}, {ex['subset']}, {self.config.name}")

            assert len(
                set(
                    [
                        len(ex["questions_no_options"]),
                        len(ex["questions"]),
                        len(ex["answers"]),
                        len(ex["answer_letter"]),
                        len(ex["options"]),
                    ]
                )
            ) == 1

            images = [
                os.path.join(image_dir, image["path"])
                for image in row["images"]
            ]

            if len(images) == 1 and len(ex["questions"]) == 1:
                continue

            options = ex["options"]
            empty_options = [i for i in range(len(options)) if len(options[i]) == 0]

            if len(empty_options) == len(options):
                continue
            elif len(empty_options) > 0:
                options = [options[i] for i in range(len(options)) if i not in empty_options]
                questions_no_options = [ex["questions_no_options"][i] for i in range(len(ex["questions_no_options"])) if i not in empty_options]
                questions = [ex["questions"][i] for i in range(len(ex["questions"])) if i not in empty_options]
                answers = [ex["answers"][i] for i in range(len(ex["answers"])) if i not in empty_options]
                answer_letter = [ex["answer_letter"][i] for i in range(len(ex["answer_letter"])) if i not in empty_options]
            else:
                questions_no_options = ex["questions_no_options"]
                questions = ex["questions"]
                answers = ex["answers"]
                answer_letter = ex["answer_letter"]
            
            questions_no_options, options = fix_options(questions, questions_no_options, options)

            choice_idx = [ord(c) - ord("A") for c in answer_letter]
            if yes_or_no:
                oe_question = questions_no_options
                direct_answer = [options[ord(c) - ord("A")] for options, c in zip(options, answer_letter)]
            else:
                oe_question = questions
                direct_answer = answers
            
            mc_question = questions_no_options
            if remove_placeholder:
                mc_question = [remove_images(q) for q in mc_question]
                oe_question = [remove_images(q) for q in oe_question]
            else:
                mc_question = [replace_images(q) for q in mc_question]
                oe_question = [replace_images(q) for q in oe_question]

            yield example_id, dict(
                subset=row["source"],
                example_id=example_id,
                images=images,
                mc_question=mc_question,
                oe_question=oe_question,
                direct_answer=direct_answer,  # assume only single answer string for each question
                choices=options,
                correct_choice_idx=choice_idx,
            )