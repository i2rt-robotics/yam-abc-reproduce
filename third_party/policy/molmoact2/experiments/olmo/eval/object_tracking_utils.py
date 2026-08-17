import difflib
import json
import random
import re
import ast
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from typing_extensions import TypedDict
import numpy as np

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from functools import partial

from dataclasses import dataclass

from olmo.models.molmo.data_formatter import seconds_to_timestamp
from olmo.util import get_absolute_coordinates
from olmo.data.video_object_tracking_datasets import PointTrajectoryEntry

import logging

log = logging.getLogger(__name__)


def format_time(time_value, format="seconds"):
    """
    TODO [QUESTION]: Merge with `format_timestamps` in DataFormatter?
    Format time value for model input/output.

    Args:
        time_value: Time in various formats (string "MM:SS.FF", float, int)
        format: "seconds" -> "2.50", "timestamp" -> "00:02.50"

    Returns:
        Formatted time string
    """
    # Parse input to float if needed
    if isinstance(time_value, str):
        if ':' in time_value:  # MM:SS.FF format
            try:
                time_obj = datetime.strptime(time_value, "%M:%S.%f")
                time_value = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            except ValueError:
                return time_value  # Return as-is if can't parse
        else:
            time_value = float(time_value)

    time_value = float(time_value)

    if format == "seconds": # e.g. "2.50"
        return f"{time_value:.2f}"
    elif format == "timestamp": # e.g. "00:02.50"
        return seconds_to_timestamp(time_value)
    else:
        raise ValueError(f"Unknown format: {format}")


format_timestamp = partial(format_time, format="timestamp")


class ObjectTrackingParser:
    """
    Parses model output text for point tracking tasks.
    Supports multiple formats:
    - video_point_track_per_frame
    - video_point_ground_start_end
    - single_point_track_per_frame
    
    Converts to standardized trajectory format for evaluation.
    """

    SUPPORTED_FORMATS = [
        'video_point_track_per_frame',
        'video_point_ground_start_end',
        'single_point_track_per_frame'
    ]

    @staticmethod
    def get_supported_formats() -> List[str]:
        return ObjectTrackingParser.SUPPORTED_FORMATS
    
    @staticmethod
    def parse_video_point_track_per_frame(text, width, height, video_fps):
        return parse_tracking_prediction_to_point_trajectory(
            text, width, height, video_fps
        )
    
    @staticmethod
    def parse_video_point_ground_start_end(text, width, height, video_fps):
        return parse_grounding_prediction_to_point_trajectory(
            text, width, height, video_fps
        )
    
    @staticmethod
    def parse_video_single_point_track_per_frame(text, width, height, video_fps):
        return parse_single_point_prediction_to_point_trajectory(
            text, width, height, video_fps
        )
    
    @classmethod
    def parse_prediction(cls, text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
        f"""
        Parse model prediction text to standardized trajectory format.
        Supported prediction types:
        - video_point_track_per_frame
        - video_point_ground_start_end
        - single_point_track_per_frame
        
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
            format = cls.detect_format(text)
            log.info(f"Auto-detected prediction format: {format}")

        if format == 'video_point_track_per_frame':
            return cls.parse_video_point_track_per_frame(text, width, height, video_fps)
        elif format == 'video_point_ground_start_end':
            return cls.parse_video_point_ground_start_end(text, width, height, video_fps)
        elif format == 'single_point_track_per_frame':
            return cls.parse_video_single_point_track_per_frame(text, width, height, video_fps)
        else:
            raise ValueError(f"Unhandled prediction_type: {format}")

    @staticmethod
    def detect_format(text) -> str:
        """ Heuristic to detect format of the prediction text."""
        if re.search(r'time\s+\d+\.?\d*', text):
            return 'video_point_track_per_frame'
        elif re.search(r'\d+:\s*\(\s*\[.*?\]\s*,\s*\[.*?\]\s*\)', text):
            return 'video_point_ground_start_end'
        elif re.search(r'\[\s*[\d.-]+\s*,\s*[\d.-]+\s*,\s*[\d.-]+\s*\]', text):
            return 'single_point_track_per_frame'
        else:
            raise ValueError("Could not detect format of prediction text.")

def extract_video_point_track_per_frame(text, width, height) -> List[Dict]:
    """
    Extract points from video_point_track_per_frame prompt style.
    Text format example:
        time 0.50
        {0: [34.0, 63.0], 1: [72.0, 49.0]}
        time 1.00
        {0: [35.0, 64.0]}
    
    Returns:
        List of timestamped points
        [{'time': 0.5, 'points': [{'id': 0, 'point': [x, y]}, ...]}, ...]
    """
    # Pattern to match "time X.XX" followed by JSON-like data
    timestamp_pattern = r'time\s+(\d+\.?\d*)\s*\n\s*(\{[^}]+\})'
    
    result = []
    
    for match in re.finditer(timestamp_pattern, text, re.MULTILINE):
        timestamp_str = match.group(1).strip()
        json_content = match.group(2).strip()
        
        try:
            seconds = float(timestamp_str)
        except ValueError:
            continue
            
        # Parse the JSON-like content
        points = []
        try:
            object_points = ast.literal_eval(json_content)
            for obj_id, coords in object_points.items():
                if len(coords) != 2:
                    continue
                x, y = coords
                if np.max([x, y]) > 100:
                    continue
                    
                # Convert from normalized coordinates to pixel coordinates
                point = get_absolute_coordinates([x, y], width, height)
                points.append({
                    'id': int(obj_id),
                    'point': point
                })
                
        except (ValueError, SyntaxError):
            continue
            
        if points:
            result.append({
                'time': seconds,
                'points': points
            })
    
    return result

def extract_video_points_start_end(text, width, height):
    """
    Extract points from video_point_ground_start_end format prompt style.
    Text format example:
        0: ([34.0, 63.0, 0.50], [35.0, 64.0, 1.00])
        1: ([72.0, 49.0, 0.50], [73.0, 50.0, 1.00])
    
    Returns:
        List of dicts with 'object_id', 'start' and 'end' keys.
        Example:
        [{
            'object_id': 0,
            'start': {'time': 0.50, 'point': np.array([x, y])},
            'end': {'time': 1.00, 'point': np.array([x, y])}
        }, ...]
    """
    result = []
    
    # Pattern matching: ID: ([x, y, time], [x, y, time])
    pattern = r'(\d+):\s*\(\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\)'
    
    for match in re.finditer(pattern, text):
        try:
            obj_id = int(match.group(1))
            start_data = [float(x.strip()) for x in match.group(2).split(',')]
            end_data = [float(x.strip()) for x in match.group(3).split(',')]
            
            if len(start_data) != 3 or len(end_data) != 3:
                continue
                
            start_x, start_y, start_time = start_data
            end_x, end_y, end_time = end_data
            
            # Validate coordinates
            if np.max([start_x, start_y, end_x, end_y]) > 100:
                continue
                
            # Convert from normalized coordinates to pixel coordinates
            start_point = get_absolute_coordinates([start_x, start_y], width, height)
            end_point = get_absolute_coordinates([end_x, end_y], width, height)
            
            result.append({
                'object_id': obj_id,
                'start': {
                    'time': start_time,
                    'point': start_point
                },
                'end': {
                    'time': end_time,
                    'point': end_point
                }
            })
            
        except (ValueError, IndexError):
            continue
    
    return result

def extract_single_point_track_per_frame(text, width, height) -> List[Dict]:
    """
    Parse single_point_track_per_frame model output.
    
    Expected format matches data_formatter.py:
    "[x, y, time], [x, y, time], [x, y, time]"
    
    Args:
        text: Model output text
        
    Returns:
        List of dicts with 'time', 'point', and 'occluded' keys
    """
    result = []
    
    # Pattern for "[x, y, time]" format
    pattern = r'\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            x = float(match[0])
            y = float(match[1])
            time_sec = float(match[2])
            
            if np.max([x, y]) > 100:
                continue
            
            # Convert from normalized coordinates to pixel coordinates
            point = get_absolute_coordinates([x, y], width, height)

            result.append({
                'time': time_sec,
                'points': [{
                    'id': 0,  # Single point tracking, so ID is always 0
                    'point': point
                }]
            })
        except ValueError:
            continue

    return result

def convert_point_tracking_to_trajectory_format(
    points_data: List[Dict], video_fps: int
) -> List[PointTrajectoryEntry]:
    """
    Convert extracted point tracking data to standardized trajectory format.
    
    Args:
        points_data: List of dicts with 'time' and 'points' keys
        video_fps: Video FPS for frame calculation
        
    Returns:
        List of PointTrajectoryEntry dicts
    """
    trajectory_data = []
    
    for entry in points_data:
        time_seconds = entry['time']
        points = entry['points']
        
        # Calculate frame index from timestamp
        frame_idx = round(time_seconds * video_fps)
        
        # Convert points to expected format
        points_dict = {}
        for point_entry in points:
            point_id = point_entry['id']
            point_coords = point_entry['point']
            points_dict[point_id] = {
                'point': point_coords.tolist() if hasattr(point_coords, 'tolist') else point_coords,
                'occluded': point_entry.get('occluded', False)
            }
        
        trajectory_entry = {
            'frame': frame_idx,
            'time': format_time(time_seconds, format="seconds"),
            'points': points_dict
        }
        trajectory_data.append(trajectory_entry)
    
    return trajectory_data

def parse_tracking_prediction_to_point_trajectory(
    prediction_text: str, video_width: int, video_height: int, video_fps: int
) -> List[PointTrajectoryEntry]:
    """
    Extracts point coordinates and timestamps from raw text predictions and
    calculates frame indices based on video FPS.
    
    Converts into standardized trajectory format to be used by video point tracking evaluators.

    Args:
        prediction_text: Predicted text for video_point_track_per_frame output
        video_width: Width of the video
        video_height: Height of the video  
        video_fps: Video FPS for frame calculation
        
    Returns:
        Structured trajectory data as list of frame dictionaries:                                                                                                │ │
        [{
            'frame': int,           # Frame index calculated from timestamp
            'time': str,            # Timestamp in MM:SS.FF format
            'points': {             # Dictionary of point_id -> point_data
                point_id: {         # Unique ID for each point
                    'point': [x, y],    # Pixel coordinates as list
                    'occluded': bool    # Occlusion status (default False)
                }
            }
        }]a
    """
    # Extract points using existing utility
    # TODO: unify prediction extraction logic into single function
    # extracted_points_data: [{'time': 0.0, 'points': [{'id': 0, 'point': array([718.07, 138.27 ])}]}]
    extracted_points_data = extract_video_point_track_per_frame(prediction_text, video_width, video_height)
    trajectory_data = convert_point_tracking_to_trajectory_format(extracted_points_data, video_fps)
    return trajectory_data
    

def parse_grounding_prediction_to_point_trajectory(prediction_text: str,
                                        video_width: int, video_height: int,
                                        video_fps: int) -> List[Dict]:
    """
    Parse point_ground_start_end model output.
    
    Expected format matches data_formatter.py:
        "0: ([34.0, 63.0, 0.50], [35.0, 64.0, 1.00])"
    
    Args:
        prediction_text: Raw prediction text with start/end points and timestamps
        video_width: Width of the video
        video_height: Height of the video  
        video_fps: Video FPS for frame calculation
        
    Returns:
        Structured trajectory data as list of frame dictionaries:                                                                                                │ │
        [{
            'frame': int,           # Frame index calculated from timestamp
            'time': str,            # Timestamp in MM:SS.FF format
            'points': {             # Dictionary of point_id -> point_data
                point_id: {         # Unique ID for each point
                    'point': [x, y],    # Pixel coordinates as list
                    'occluded': bool    # Occlusion status (default False)
                }
            }
        }]
    """
     
    # [{'object_id': '0', 'start': {'time': 0, 'point': [0.5, 0.5]}, 'end': {'time': 5, 'point': [0.6, 0.6]}}]
    grounding_data = extract_video_points_start_end(prediction_text, video_width, video_height)

    entry_per_frame = {} # {frame_idx: {id: point, ...}, ...} # Used for adding points since points_data is point-centric
    trajectory_data = [] # per frame data

    for entry in grounding_data:
        start_time = entry['start']['time']
        start_point = entry['start']['point']
        start_frame = int(start_time * video_fps)

        end_time = entry['end']['time']
        end_point = entry['end']['point']
        end_frame = int(end_time * video_fps)

        # Add start point
        if start_frame not in entry_per_frame:
            entry_per_frame[start_frame] = {}
        entry_per_frame[start_frame][entry['object_id']] = start_point

        # Add end point
        if end_frame not in entry_per_frame:
            entry_per_frame[end_frame] = {}
        entry_per_frame[end_frame][entry['object_id']] = end_point

    # Convert to trajectory format
    for frame_idx, points in entry_per_frame.items():
        time_in_seconds = frame_idx / video_fps
        trajectory_data.append({
            'frame': frame_idx,
            'time': f"{int(time_in_seconds//60):02}:{int(time_in_seconds%60):02}.{int((time_in_seconds*100)%100):02}",
            'points': {
                object_id: {
                    'point': point.tolist() if hasattr(point, 'tolist') else point,
                    'occluded': False
                } for object_id, point in points.items()
            }
        })

    return trajectory_data

def parse_single_point_prediction_to_point_trajectory(prediction_text: str, 
                                                      video_width: int, video_height: int, 
                                                      video_fps: int) -> List[Dict]:
    """
    Parse single_point_track_per_frame model output.
    
    Expected format matches data_formatter.py:
    "[x, y, time], [x, y, time], [x, y, time]"

    Args:
        prediction_text: Raw prediction text with x, y, time entries
        video_width: Width of the video
        video_height: Height of the video  
        video_fps: Video FPS for frame calculation
        
    Returns:
        Structured trajectory data as list of frame dictionaries:                                                                                                │ │
        [{
            'frame': int,           # Frame index calculated from timestamp
            'time': str,            # Timestamp in MM:SS.FF format
            'points': {             # Dictionary of point_id -> point_data
                point_id: {         # Unique ID for each point
                    'point': [x, y],    # Pixel coordinates as list
                    'occluded': bool    # Occlusion status (default False)
                }
            }
        }]
    """
    
    # Extract points from ["x, y, time"] data
    points_data = extract_single_point_track_per_frame(prediction_text, video_width, video_height)
    trajectory_data = convert_point_tracking_to_trajectory_format(points_data, video_fps)

    return trajectory_data

def ann_to_mask(mask_ann) -> np.ndarray:
    """
    Decodes RLE masks to binary mask
    """
    from pycocotools import mask as mask_utils
    if isinstance(mask_ann, np.ndarray):
        return mask_ann

    if isinstance(mask_ann, list): # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        h,w = mask_ann[0]['size']
        rles = mask_utils.frPyObjects(mask_ann, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(mask_ann['counts'], list): # uncompressed RLE
        h, w = mask_ann['size']
        rle = mask_utils.frPyObjects(mask_ann, h, w)
    else: # compressed RLE
        rle = mask_ann
    mask = mask_utils.decode(rle)
    return mask

def is_point_in_region(point: Tuple[float, float], mask: np.ndarray) -> bool:
    """
    Check if point falls within the segmentation mask region.
    (Copied from olmo.eval.evaluators.py)
    """
    height, width = mask.shape
    x, y = point
    
    # Round coordinates to nearest integer
    x_int = int(round(x))
    y_int = int(round(y))
    
    # Check bounds
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False
    
    # Check if point is within region
    return mask[y_int, x_int]

def compute_precision(row_ind: np.ndarray, col_ind: np.ndarray, 
                     preds: np.ndarray, masks: List[np.ndarray]) -> float:
    """
    Compute precision: correctly placed points / total predicted points.
    (Adapted from olmo.eval.evaluators.py)
    """
    if len(preds) == 0:
        return 1.0  # No predictions made, perfect precision
        
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(preds)


def compute_recall(row_ind: np.ndarray, col_ind: np.ndarray,
                  preds: np.ndarray, masks: List[np.ndarray]) -> float:
    """
    Compute recall: correctly placed points / total ground truth points.
    (Adapted from olmo.eval.evaluators.py)
    """
    if len(masks) == 0:
        return 1.0  # No ground truth to miss
        
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(masks)


def f1_score(precision: float, recall: float, epsilon: float = 1e-10) -> float:
    """
    Compute F1 score from precision and recall.
    (Copied from olmo.eval.evaluators.py)
    """
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + epsilon)

def evaluate_frame_predictions(pred_points: List[Tuple[float, float]], 
                             gt_points: List[Tuple[float, float]],
                             masks: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Evaluate predictions for a single frame.
    
    Args:
        pred_points: List of predicted (x, y) coordinates
        gt_points: List of ground truth (x, y) coordinates  
        masks: List of segmentation masks (one per GT point)
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Handle edge cases
    if len(gt_points) == 0:
        # No ground truth - perfect score if no predictions, zero if any predictions
        precision = recall = f1 = float(len(pred_points) == 0)
        return precision, recall, f1
        
    if len(pred_points) == 0:
        # No predictions made - zero precision and recall
        return 0.0, 0.0, 0.0
    
    # Convert to numpy arrays for distance calculation
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    
    # Compute pairwise distances and find optimal assignment
    distances = cdist(pred_points, gt_points)
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Compute metrics based on mask overlap
    precision = compute_precision(row_ind, col_ind, pred_points, masks)
    recall = compute_recall(row_ind, col_ind, pred_points, masks)
    f1 = f1_score(precision, recall)
    
    return precision, recall, f1

def load_frame_segmentation_masks(mask_item: Dict, frame_idx: int) -> List[np.ndarray]:
    """
    Get all binary masks for frame from HF mask_item.
    
    Args:
        mask_item: Item from HuggingFace dataset
        frame_idx: Frame index to get masks for
        
    Returns:
        List of binary masks (one per object)
    """
    def find_first_mask(mask_list):
        for m in mask_list:
            if m is not None:
                return m
        return None
    
    h, w = mask_item['h'], mask_item['w']
    empty_mask = np.zeros((h, w), dtype=bool)

    masks = []
    
    # Get mask data for each object
    for mask_id in mask_item['mask_id']:
        mask_list = mask_item['masks'][mask_id]  # List of masks for this object

        first_mask = find_first_mask(mask_list)
        if first_mask is None: # No masks available for this object
            masks.append(empty_mask)
            continue
        
        binary_mask = empty_mask

        # sometimes masks are not annotated for all frames, e.g. burst-test (annotated at 1 fps only)
        # if frame_idx exists in mask_list, use it to align the frames, instead of assuming 1:1 mapping
        if 'frame' in first_mask: # frame index exists so use it to find the right mask
            for mask in mask_list:
                if mask['mask'] is not None and mask['frame'] <= frame_idx:
                    if mask['frame'] == frame_idx:
                        binary_mask = ann_to_mask(mask['mask'])
                        break
        else: # assume 1:1 mapping of mask_list to frames
            if frame_idx < len(mask_list) and mask_list[frame_idx] is not None:
                binary_mask = ann_to_mask(mask_list[frame_idx])

        binary_mask = binary_mask.astype(bool)
        masks.append(binary_mask)
    
    return masks

def evaluate_video_point_tracking_with_masks(
    pred_trajectory: List[Dict] | List[PointTrajectoryEntry], 
    gt_trajectory: List[Dict] | List[PointTrajectoryEntry],
    mask_item: Dict) -> Dict[str, float]:
    """
    Evaluate predictions for a single video.
    
    Args:
        pred_trajectory: Predicted trajectory data
            Format:
            [{
                'frame': int,           # Frame index
                'time': str,            # Timestamp in MM:SS.FF format
                'points': {             # Dictionary of point_id -> point_data
                    point_id: {         # Unique ID for each point
                        'point': [x, y],    # Pixel coordinates as list
                        'occluded': bool    # Occlusion status (default False)
                    }
                }
            }]

        gt_trajectory: Ground truth trajectory data
        mask_item: Mask item containing segmentation masks for the video
            Format:
            {
                "mask_id": List of mask IDs,
                "masks": {
                    "mask_id_1": List of masks for object 1,
                    "mask_id_2": List of masks for object 2,
                    ...
                },
                "h": Height of the video,
                "w": Width of the video
            }
        
    Returns:
        Dictionary of aggregated metrics:
        {
            'precision': float,
            'recall': float,
            'f1': float,
            'num_frames': int,
            'frames_with_pred': int,
            'frames_with_gt': int,
            'frames_with_both': int,
            'frames_pred_only': int,
            'frames_gt_only': int,
            'frame_details': List of per-frame metrics for debugging
        }
    """
    # Create frame-indexed lookup for easier matching
    pred_by_frame = {entry['frame']: entry for entry in pred_trajectory}
    gt_by_frame = {entry['frame']: entry for entry in gt_trajectory}
    
    # Find all frames that have either predictions OR ground truth
    all_frames = set(pred_by_frame.keys()) | set(gt_by_frame.keys())
    
    if not all_frames:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_frames': 0}
    
    frame_metrics = []
    for frame_idx in sorted(all_frames):
        # Get prediction and GT data for this frame (may be None if frame not present)
        pred_frame = pred_by_frame.get(frame_idx)
        gt_frame = gt_by_frame.get(frame_idx)
        
        # IF no predictions and no ground truth, skip this frame
        if pred_frame is None and gt_frame is None:
            continue
        
        # Extract predicted points (convert from dict format)
        pred_points = []
        if pred_frame is not None:
            for obj_id, point_data in pred_frame['points'].items():
                pred_points.append(tuple(point_data['point']))
        
        # Extract ground truth points and get object IDs for mask loading
        gt_points = []
        object_ids = []
        if gt_frame is not None:
            for obj_id, point_data in gt_frame['points'].items():
                gt_points.append(tuple(point_data['point']))
                object_ids.append(str(obj_id))
        
        masks = load_frame_segmentation_masks(mask_item, frame_idx)
        
        # Evaluate this frame
        precision, recall, f1 = evaluate_frame_predictions(pred_points, gt_points, masks)
        frame_metrics.append({
            'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'frame_idx': frame_idx,
            'has_pred': pred_frame is not None,
            'has_gt': gt_frame is not None,
            'num_pred_points': len(pred_points),
            'num_gt_points': len(gt_points)
        })
    
    # Average metrics across frames for overall video score
    avg_precision = np.mean([m['precision'] for m in frame_metrics])
    avg_recall = np.mean([m['recall'] for m in frame_metrics])
    avg_f1 = np.mean([m['f1'] for m in frame_metrics])
    
    # Calculate diagnostic statistics
    frames_with_pred = sum(1 for m in frame_metrics if m['has_pred'])
    frames_with_gt = sum(1 for m in frame_metrics if m['has_gt'])
    frames_with_both = sum(1 for m in frame_metrics if m['has_pred'] and m['has_gt'])
    frames_pred_only = sum(1 for m in frame_metrics if m['has_pred'] and not m['has_gt'])
    frames_gt_only = sum(1 for m in frame_metrics if not m['has_pred'] and m['has_gt'])
    
    return {
        'precision': avg_precision,
        'recall': avg_recall, 
        'f1': avg_f1,
        'num_frames': len(frame_metrics),
        'frames_with_pred': frames_with_pred,
        'frames_with_gt': frames_with_gt,
        'frames_with_both': frames_with_both,
        'frames_pred_only': frames_pred_only,
        'frames_gt_only': frames_gt_only,
        'frame_details': frame_metrics  # For debugging
    }

class HOTAMetric:
    """HOTA metric for multi-object tracking with pointing predictions."""
    
    def __init__(self, alpha_thresholds=None):
        if alpha_thresholds is None:
            self.alpha_thresholds = np.arange(0.05, 1.0, 0.05)
        else:
            self.alpha_thresholds = alpha_thresholds
    
    def prepare_data_for_hota(self, pred_trajectory, gt_trajectory, mask_item):
        """
        Convert trajectory data to HOTA format.
        
        Returns:
            data: dict with keys needed for HOTA computation
        """
        # Create frame-indexed lookups
        pred_by_frame = {entry['frame']: entry for entry in pred_trajectory}
        gt_by_frame = {entry['frame']: entry for entry in gt_trajectory}
        
        # Get all unique object IDs
        all_gt_ids = set()
        all_pred_ids = set()
        for entry in gt_trajectory:
            all_gt_ids.update(entry['points'].keys())
        for entry in pred_trajectory:
            all_pred_ids.update(entry['points'].keys())
        
        # Create ID mappings to continuous integers
        gt_id_to_idx = {str(obj_id): idx for idx, obj_id in enumerate(sorted(all_gt_ids))}
        pred_id_to_idx = {str(obj_id): idx for idx, obj_id in enumerate(sorted(all_pred_ids))}
        
        # Get all frames
        all_frames = sorted(set(pred_by_frame.keys()) | set(gt_by_frame.keys()))
        
        data = {
            'num_gt_ids': len(all_gt_ids),
            'num_tracker_ids': len(all_pred_ids),
            'num_timesteps': len(all_frames),
            'gt_ids': [],
            'tracker_ids': [],
            'similarity_scores': [],
            'num_gt_dets': 0,
            'num_tracker_dets': 0,
        }
        
        # Process each frame
        for frame_idx in all_frames:
            pred_frame = pred_by_frame.get(frame_idx)
            gt_frame = gt_by_frame.get(frame_idx)
            
            # Get GT data
            gt_ids_list = []
            gt_points_list = []
            if gt_frame is not None:
                for obj_id, point_data in sorted(gt_frame['points'].items()):
                    gt_ids_list.append(gt_id_to_idx[str(obj_id)])
                    gt_points_list.append(point_data['point'])
            
            # Get prediction data
            pred_ids_list = []
            pred_points_list = []
            if pred_frame is not None:
                for obj_id, point_data in sorted(pred_frame['points'].items()):
                    pred_ids_list.append(pred_id_to_idx[str(obj_id)])
                    pred_points_list.append(point_data['point'])
            
            # Convert to numpy arrays
            gt_ids_t = np.array(gt_ids_list, dtype=int)
            pred_ids_t = np.array(pred_ids_list, dtype=int)
            
            data['gt_ids'].append(gt_ids_t)
            data['tracker_ids'].append(pred_ids_t)
            data['num_gt_dets'] += len(gt_ids_t)
            data['num_tracker_dets'] += len(pred_ids_t)
            
            # Compute similarity matrix
            similarity = self._compute_similarity_matrix(
                pred_points_list, gt_points_list, mask_item, frame_idx
            )
            data['similarity_scores'].append(similarity)
        
        return data
    
    def _compute_similarity_matrix(self, pred_points, gt_points, mask_item, frame_idx):
        """
        Compute similarity between predictions and GT for one frame.
        Similarity = 1 if point in mask, 0 otherwise.
        
        Returns:
            similarity: (num_gt, num_pred) matrix
        """
        if len(pred_points) == 0 or len(gt_points) == 0:
            return np.zeros((len(gt_points), len(pred_points)))
        
        # Load masks for this frame
        masks = load_frame_segmentation_masks(mask_item, frame_idx)
        
        similarity = np.zeros((len(gt_points), len(pred_points)))
        
        for i, (gt_point, mask) in enumerate(zip(gt_points, masks)):
            for j, pred_point in enumerate(pred_points):
                # Binary similarity: 1 if point in mask, 0 otherwise
                if is_point_in_region(pred_point, mask):
                    similarity[i, j] = 1.0
        
        return similarity
    
    def compute_hota(self, data):
        """
        Compute HOTA metrics given prepared data.
        Adapted from TrackEval HOTA implementation.
        """

        num_alphas = len(self.alpha_thresholds)
        
        # Initialize results
        res = {
            'HOTA_TP': np.zeros(len(self.alpha_thresholds)),
            'HOTA_FN': np.zeros(len(self.alpha_thresholds)),
            'HOTA_FP': np.zeros(len(self.alpha_thresholds)),
            'HOTA': np.zeros(len(self.alpha_thresholds)),
            'DetA': np.zeros(len(self.alpha_thresholds)),
            'AssA': np.zeros(len(self.alpha_thresholds)),
            'DetRe': np.zeros(len(self.alpha_thresholds)),
            'DetPr': np.zeros(len(self.alpha_thresholds)),
            'LocA': np.zeros(len(self.alpha_thresholds)),
        }

        if data['num_tracker_dets'] == 0 and data['num_gt_dets'] == 0:
            return {k: np.ones(num_alphas) for k in res.keys()}
        
        # No predictions
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = np.full(num_alphas, data['num_gt_dets'])
            res['LocA'] = np.ones(num_alphas)  # Placeholder (DetA=0 anyway)
            # HOTA, DetA, AssA stay at 0
            return res

        # No GT
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = np.full(num_alphas, data['num_tracker_dets'])
            res['LocA'] = np.ones(num_alphas)  # Placeholder (DetA=0 anyway)
            # HOTA, DetA, AssA stay at 0
            return res
        
        # Variables for global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))
        
        # First pass: accumulate global track information
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            
            # Compute normalized similarity (IoU-like)
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
            
            # Count detections per ID
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1
        
        # Global alignment score (learned ID correspondence)
        global_alignment_score = potential_matches_count / (
            gt_id_count + tracker_id_count - potential_matches_count + np.finfo('float').eps
        )
        
        # Track matches for each alpha threshold
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.alpha_thresholds]
        
        # Second pass: compute per-frame scores
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                for a in range(len(self.alpha_thresholds)):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a in range(len(self.alpha_thresholds)):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue
            
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
            
            # Hungarian matching
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            
            # Compute statistics for each alpha threshold
            for a, alpha in enumerate(self.alpha_thresholds):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1
        
        # Compute association scores
        for a in range(len(self.alpha_thresholds)):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
        
        # Compute final metrics
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        
        return res


# Updated evaluation function
def evaluate_video_point_tracking_with_hota(
    pred_trajectory: List[Dict],
    gt_trajectory: List[Dict],
    mask_item: Dict
) -> Dict[str, float]:
    """
    Evaluate predictions with both spatial metrics (P/R/F1) and tracking metrics (HOTA).
    """
    # Compute existing spatial metrics
    spatial_metrics = evaluate_video_point_tracking_with_masks(
        pred_trajectory, gt_trajectory, mask_item
    )
    
    # Compute HOTA tracking metrics
    hota_metric = HOTAMetric()
    hota_data = hota_metric.prepare_data_for_hota(pred_trajectory, gt_trajectory, mask_item)
    hota_results = hota_metric.compute_hota(hota_data)
    
    # Combine results (use alpha=0.5 threshold for main metrics)
    alpha_05_idx = 9  # 0.5 is at index 9 in range(0.05, 1.0, 0.05)
    # doesn't really matter because it's just 0 or 1 for point-in-mask
    
    return {
        # Spatial metrics (per-frame detection quality)
        'precision': spatial_metrics['precision'],
        'recall': spatial_metrics['recall'],
        'f1': spatial_metrics['f1'],
        
        # HOTA tracking metrics (ID consistency)
        'HOTA': float(hota_results['HOTA'][alpha_05_idx]),
        'DetA': float(hota_results['DetA'][alpha_05_idx]),
        'AssA': float(hota_results['AssA'][alpha_05_idx]),
        'DetPr': float(hota_results['DetPr'][alpha_05_idx]),
        'DetRe': float(hota_results['DetRe'][alpha_05_idx]),
        'LocA': float(hota_results['LocA'][alpha_05_idx]),
        
        # Full HOTA arrays (all alpha thresholds)
        'HOTA_array': hota_results['HOTA'].tolist(),
        'DetA_array': hota_results['DetA'].tolist(),
        'AssA_array': hota_results['AssA'].tolist(),
        
        # Diagnostic info
        'num_frames': spatial_metrics['num_frames'],
        'frames_with_pred': spatial_metrics['frames_with_pred'],
        'frames_with_gt': spatial_metrics['frames_with_gt'],
    }