import numpy as np
from functools import partial


def fuzzy_matching(pred: str) -> str:
    return pred.split(' ')[0].rstrip('.').strip()


def extact_match(pred: str, target: str) -> bool:
    return 1. if pred.lower() == target.lower() else 0.


def abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target


def to_float(pred: str) -> float | None:
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float,
    end: float,
    interval: float,
) -> float:
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def vsi_bench_na_score(
    pred: str,
    target: str,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    try:
        score = mean_relative_accuracy(
            to_float(fuzzy_matching(pred)),
            to_float(target),
            start,
            end,
            interval,
        )
    except:
        score = 0.
    return score