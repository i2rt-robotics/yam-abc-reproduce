import json
import os
from pathlib import Path
import re
from typing import List, Optional, Dict, Any, Tuple
import string

import pandas as pd
import datasets
from cached_path import cached_path



class MultiImageVLM3RBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    ANNOTATION_FILE = "multi-img_vlm3r_4frames.json"
    _OPT_LINE_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$")

    def __init__(self, data_source: str, *args, **kwargs):
        self.data_source = data_source

        super().__init__(*args, **kwargs, dataset_name="multi_image_vlm3r")
    
    def _info(self):
        features = datasets.Features(
            {
                "example_id": datasets.Value("string"),
                "images": datasets.Sequence(datasets.Value("string")),
                "original_question": datasets.Value("string"),
                "question": datasets.Value("string"),
                "choices": datasets.Sequence(datasets.Value("string")),
                "correct_choice_idx": datasets.Value("int64"),
                "question_type": datasets.Value("string"),
                "scene_name": datasets.Value("string"),
                "data_source": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(features=features)
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        gen_kwargs = {}
        base_dir = os.path.join(self.data_source, "multi_image_datasets", "3D")
        for split_name in ["train"]:
            split_gen_kwargs = {}

            json_path = os.path.join(
                base_dir, "VLM-3R", self.ANNOTATION_FILE,
            )
            split_gen_kwargs["json_path"] = json_path
            split_gen_kwargs["prev_image_dir"] = os.path.join(
                self.data_source, "multi_image_data",
            )
            split_gen_kwargs["image_dir"] = base_dir
            gen_kwargs[split_name] = split_gen_kwargs
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=gen_kwargs["train"]),
        ]

    def _parse_qa(self, question_text: str, answer: str) -> Optional[Tuple[str, List[str], int]]:
        """
        Extract options listed under a line containing 'Options:' or anywhere in the text.
        Returns:
        - the question text right before 'Options:',
        - the option TEXT (sorted by letter A..Z if letters present).
        - the correct choice index.
        """
        # Normalize answer to a single letter (first alpha char)
        m_ans = re.search(r'[A-Za-z]', answer or '')
        if not m_ans:
            return None
        ans_letter = m_ans.group(0).upper()

         # Locate "Options:" line
        lines = question_text.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines) if "options:" in ln.lower())
        except StopIteration:
            start = -1
        
        if start <= 0:
            return None
        qtext = "\n".join(lines[:start]).strip()
        region = lines[start + 1:]

        # Collect (letter, text) tuples
        found: List[Tuple[str, str]] = []
        for ln in region:
            m = self._OPT_LINE_RE.match(ln)
            if m:
                letter, text = m.group(1).strip(), m.group(2).strip()
                found.append((letter, text))

        if found and all(len(t[0]) == 1 and "A" <= t[0] <= "Z" for t in found):
            letters = [t[0] for t in found]
            if ans_letter not in letters:
                return None
            
            # Sort options by letter; compute index by locating the answer letter post-sort
            found_sorted = sorted(found, key=lambda x: x[0])
            option_texts_sorted = [t[1] for t in found_sorted]
            correct_choice_idx = [t[0] for t in found_sorted].index(ans_letter)
            
            return qtext, option_texts_sorted, correct_choice_idx
        else:
            return None
    
    def _generate_examples(self, json_path: str, prev_image_dir: str, image_dir: str):

        # Load raw JSON
        with open(cached_path(json_path), "r") as f:
            raw: Dict[str, Dict[str, Any]] = json.load(f)
        
        # Yield standardized examples from raw data
        for ex_id, ex in raw.items():
            paths: List[str] = ex["paths"]
            if len(paths) <= 1:
                continue
            # Fix image paths
            paths = [path.replace(prev_image_dir, image_dir) for path in paths]

            if (ret := self._parse_qa(ex["question"], ex["answer"])) is None:
                continue
            qtext, options, correct_choice_idx = ret
            question_type = ex.get("question_type", "unknown")

            yield ex_id, dict(
                example_id=ex_id,
                images=paths,
                original_question=ex["question"],
                question=qtext,
                choices=options,
                correct_choice_idx=correct_choice_idx,
                question_type=question_type,
                scene_name=ex["scene_name"],
                data_source=ex["data_source"],
            )