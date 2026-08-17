# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Evaluation utilites for point tracking.
Different from object tracking which marks as correct if within segmentation mask,
point tracking requires the predicted point to be within a certain pixel distance.

Prediction should also include occlusion prediction even if point is in the view.
"""

import ast
import csv
import functools
import io
import os
from os import path
import re
import pickle
import random
import torch
import numpy as np
from typing import Dict, List, Iterable, Mapping, Optional, Tuple, Union


from olmo.util import get_absolute_coordinates
from olmo.data.video_object_tracking_datasets import PointTrajectoryEntry
from olmo.eval.object_tracking_utils import convert_point_tracking_to_trajectory_format

class PointTrackingParser:
    """
    Parser for point tracking prediction formats.
    """

    SUPPORTED_FORMATS = [
        'video_point_track_all_frames_with_occlusion',
    ]

    @staticmethod
    def parse_video_point_track_all_frames_with_occlusion(text, width, height, video_fps):
        return parse_tracking_prediction_with_occlusion_to_point_trajectory(
            text, width, height, video_fps
        )

    @classmethod
    def parse_prediction(cls, text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
        """
        Parse model prediction text to standardized trajectory format.
        Supported prediction types:
            - video_point_track_all_frames_with_occlusion
        
        Args:
            text: Raw prediction text
            width: Video width for coordinate conversion
            height: Video height for coordinate conversion
            video_fps: Video FPS for frame calculation
            format: Type of prediction format
        Returns:
            List of PointTrajectoryEntry dicts
        """

        if not format:
            format = 'video_point_track_all_frames_with_occlusion'

        if format == 'video_point_track_all_frames_with_occlusion':
            return cls.parse_video_point_track_all_frames_with_occlusion(text, width, height, video_fps)
        else:
            raise ValueError(f"Unhandled prediction_type: {format}")

def extract_video_point_track_per_frame_with_occlusion(
    text: str, width: int, height: int
) -> List[Dict]:
    """
    Extract points from video_point_track_all_frames_with_occlusion format.
    Text format example (yes if occluded):
        time 0.50
        {0: [52.4, 40.7], 1: [52.4, 41.5]}
        time 1.00
        {0: [52.4, 40.7, yes], 1: [52.4, 41.5], 2: [52.8, 40.5, yes]}
    
    Returns:
        List of timestamped points
        [{'time': 0.5, 'points': [{'id': 0, 'point': [x, y], 'occluded': False}, ...]}, ...]
    """
    timestamp_pattern = r'time\s+(\d+\.?\d*)\s*\n\s*(\{[^}]+\})'

    result = []

    for match in re.finditer(timestamp_pattern, text, re.MULTILINE):
        timestamp_str = match.group(1).strip()
        json_content = match.group(2).strip()

        try:
            timestamp = float(timestamp_str)
        except ValueError:
            continue
        
        points = []
        try:
            object_points = ast.literal_eval(json_content)
            for obj_id, coords in object_points.items():
                if len(coords) == 2:
                    x, y = coords
                    occluded = False
                elif len(coords) == 3:
                    x, y, occluded_str = coords
                    occluded = occluded_str.strip().lower() in ['yes', 'true', '1']
                if np.max([x, y]) > 100:
                    continue

                # Convert from normalized coordinates to pixel coordinates
                point = get_absolute_coordinates([x, y], width, height)
                points.append({
                    'id': int(obj_id),
                    'point': point,
                    'occluded': occluded
                })

        except (ValueError, SyntaxError):
            continue

        if points:
            result.append({
                'time': timestamp,
                'points': points
            })

    return result

def parse_tracking_prediction_with_occlusion_to_point_trajectory(
    prediction_text: str, video_width: int, video_height: int, video_fps: int
) -> List[PointTrajectoryEntry]:
    
    extracted_points_data = extract_video_point_track_per_frame_with_occlusion(prediction_text, video_width, video_height)
    trajectory_data = convert_point_tracking_to_trajectory_format(extracted_points_data, video_fps)

    return trajectory_data

def create_pred_tracks_and_occlusions(
    trajectory_data: List[PointTrajectoryEntry],
    num_frames: int,
    num_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert list of PointTrajectoryEntry to numpy arrays for evaluation.
    
    Args:
        trajectory_data: List of PointTrajectoryEntry dicts
        num_frames: Total number of frames in the video
        num_points: Optional fixed number of points. If None, inferred from data.
    
    Returns:
        occluded: [n, t] boolean array where True indicates occlusion
        target_points: [n, t, 2] array of [x, y] target points
    """
    all_ids = set()
    for entry in trajectory_data:
        all_ids.update(entry['points'].keys())

    if num_points is None:
        num_points = len(all_ids)

    id_to_index = {obj_id: idx for idx, obj_id in enumerate(sorted(all_ids))}
    
    occluded = np.ones((num_points, num_frames), dtype=bool)
    target_points = np.zeros((num_points, num_frames, 2), dtype=np.float32)

    for entry in trajectory_data:
        frame_idx = entry['frame']
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        for point_id, pt in entry['points'].items():
            if point_id not in id_to_index:
                continue
            point_idx = id_to_index[point_id]
            occluded[point_idx, frame_idx] = pt['occluded']
            if not pt['occluded']:
                target_points[point_idx, frame_idx] = pt['point']
    
    return occluded, target_points

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
        query_points: The query points, an in the format [t, y, x].  Its size is
            [b, n, 3], where b is the batch size and n is the number of queries
        gt_occluded: A boolean array of shape [b, n, t], where t is the number
            of frames.  True indicates that the point is occluded.
        gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
            in the format [x, y]
        pred_occluded: A boolean array of predicted occlusions, in the same
            format as gt_occluded.
        pred_tracks: An array of track predictions from your algorithm, in the
            same format as gt_tracks.
        query_mode: Either 'first' or 'strided', depending on how queries are
            sampled.  If 'first', we assume the prior knowledge that all points
            before the query point are occluded, and these are removed from the
            evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
            predicted to be within the given pixel threshold, ignoring occlusion
            prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
            threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
                np.square(pred_tracks - gt_tracks),
                axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
                is_correct & evaluation_points,
                axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
                is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

def reduce_masked_mean(input, mask, dim=None, keepdim=False, eps=1e-6):
    r"""Masked mean
    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning
    .. math::
            \text{output} =
            \frac
            {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
            {\epsilon + \sum_{i=1}^N \text{mask}_i}
    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.
    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.
    The interface is similar to `torch.mean()`.
    Args:
        input (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.
        eps (float, optional): Avoid division by zero.
    Returns:
            Tensor: mean tensor.
    """

    mask = mask.expand_as(input)
    eps = 1e-6

    prod = input * mask

    if dim is None:
            numer = torch.sum(prod)
            denom = torch.sum(mask)
    else:
            numer = torch.sum(prod, dim=dim, keepdim=keepdim)
            denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (denom + eps)
    return mean

def latex_table(mean_scalars: Mapping[str, float]) -> str:

    """Generate a latex table for displaying TAP-Vid and PCK metrics."""
    if 'average_jaccard' in mean_scalars:
        latex_fields = [
                'average_jaccard',
                'average_pts_within_thresh',
                'occlusion_accuracy',
                'jaccard_1',
                'jaccard_2',
                'jaccard_4',
                'jaccard_8',
                'jaccard_16',
                'pts_within_1',
                'pts_within_2',
                'pts_within_4',
                'pts_within_8',
                'pts_within_16',
        ]
        header = (
                'AJ & $<\\delta^{x}_{avg}$ & OA & Jac. $\\delta^{0}$ & '
                + 'Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & '
                + 'Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & $<\\delta^{0}$ & '
                + '$<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & '
                + '$<\\delta^{4}$'
        )
    else:
        latex_fields = ['PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5']
        header = ' & '.join(latex_fields)

    body = ' & '.join(
            [f'{float(np.array(mean_scalars[x]*100)):.3}' for x in latex_fields]
    )
    return '\n'.join([header, body])


def evaluate_video_point_tracking(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
):
    # Batch dimension is required for compute_tapvid_metrics
    # Add batch dimension of 1
    tapvid_metrics = compute_tapvid_metrics(
        np.expand_dims(query_points, axis=0),
        np.expand_dims(gt_occluded, axis=0),
        np.expand_dims(gt_tracks, axis=0),
        np.expand_dims(pred_occluded, axis=0),
        np.expand_dims(pred_tracks, axis=0),
        query_mode,
    )

    # Remove batch dimension
    for key, value in tapvid_metrics.items():
        tapvid_metrics[key] = value[0].item()

    return tapvid_metrics