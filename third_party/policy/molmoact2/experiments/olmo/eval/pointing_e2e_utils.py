"""
Utilities for two-stage point tracking evaluation pipeline.

Handles:
- Parsing point_ground_start_end predictions to extract grounded objects
- Creating synthetic single_point_track_per_frame prompts from grounding results
- Aggregating metrics from both stages
"""

from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from create_data.pointing.object_tracking_instruction import PromptGenerator
from olmo.util import get_absolute_coordinates
from olmo.data.video_object_tracking_datasets import ObjectTracking
from olmo.eval.object_tracking_utils import (
    parse_single_point_predictions_to_point_trajectory,
    evaluate_video_point_tracking_with_masks, format_time
)


def parse_grounding_predictions(ground_start_end_data: Dict) -> Dict[str, List[Dict]]:
    """
    Parse point_ground_start_end predictions to extract grounded objects with start points.
    Used for creating synthetic single_point_track_per_frame prompts from start points.

    Expected format in predictions:
    "0: ([76.3, 58.0, 0.00], [53.0, 56.8, 11.00])
    1: ([48.0, 56.9, 0.00], [73.1, 60.2, 11.00])
    2: ([50.8, 60.4, 0.00], [49.4, 59.4, 11.00])"

    Args:
        ground_start_end_data: Full prediction data from point_ground_start_end evaluation
        
    Returns:
        Dict mapping video_id to list of grounded objects with start/end points
    """

    grounded_objects = {}
    
    # Handle both list and dict formats
    if isinstance(ground_start_end_data, list):
        predictions = ground_start_end_data
    else:
        predictions = ground_start_end_data.get('predictions', [])
    
    for pred in predictions:
        video_id = pred.get('example_id', pred.get('id', ''))
        prediction_text = pred.get('prediction', pred.get('text', ''))
        metadata = pred.get('metadata', {})
        
        if not prediction_text:
            continue
        
        # Parse the prediction text to extract objects
        objects = []
        
        # Split by newlines and parse each line
        lines = prediction_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Pattern for the format: "0: ([x1, y1, t1], [x2, y2, t2])"
            # Allow spaces between commas: obj_id: ([start_x, start_y, start_time], [end_x, end_y, end_time])
            pattern = r'(\d+):\s*\(\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]\s*,\s*\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]\)'
            
            match = re.search(pattern, line)
            if match:
                obj_id = int(match.group(1))
                start_x = float(match.group(2))
                start_y = float(match.group(3))
                start_time = float(match.group(4))
                end_x = float(match.group(5))
                end_y = float(match.group(6))
                end_time = float(match.group(7))
                
                objects.append({
                    'obj_id': obj_id,
                    'start_time': start_time,
                    'start_point': [start_x, start_y], # Still normalized 0-100
                    'end_time': end_time,
                    'end_point': [end_x, end_y], # Still normalized 0-100
                    'expression': metadata.get('expression', 'object'),
                    'video_fps': metadata.get('video_fps', 6.0),
                    'original_metadata': metadata  # Keep original metadata for mask lookup
                })
        
        if objects:
            grounded_objects[video_id] = objects
    
    return grounded_objects

def format_single_point_track_output(
    points: List[Tuple[float, float]],
    times: List[float],
    occluded: Optional[List[bool]] = None
) -> str:
    """
    Format single_point_track_per_frame output from list of points and times.
    
    Expected format matches data_formatter.py:
    "[x, y, time], [x, y, time], [x, y, time]"
    
    Args:
        points: List of (x, y) coordinates
        times: List of timestamps
        occluded: Optional list of occlusion flags (points with occluded=True are skipped)
        
    Returns:
        Formatted string for model output
    """
    output_parts = []
    
    for i, (time, point) in enumerate(zip(times, points)):
        # Skip occluded points (matches data_formatter.py behavior)
        if occluded and i < len(occluded) and occluded[i]:
            continue
            
        x, y = point
        point_json = f'[{x:.1f}, {y:.1f}, {time}]'
        output_parts.append(point_json)
    
    return ", ".join(output_parts)

def create_single_point_tracking_examples(
    ground_start_end_data: Dict,
    grounded_objects: Dict[str, List[Dict]],
    base_task_name: str
) -> List[Dict]:
    """
    Create synthetic examples for single_point_track_per_frame evaluation.
    
    For each grounded object, create a tracking prompt starting from its initial point.
    These are just JSON examples that will be passed to the model, not a full HF dataset.
    
    Args:
        ground_start_end_data: Original grounding prediction data
            Expected with predictions key in format "object_id: ([start_x, start_y, start_time], [end_x, end_y, end_time])\n..."
            [
                {
                    "example_id": "video_1",
                    "prediction": "0: ([100, 200, 0], [150, 250, 1])\n1: ([300, 400, 0], [350, 450, 1])"
                },
                ...
            ]
        grounded_objects: Parsed grounded objects per video
        base_task_name: Original task name to extract dataset parameters
        
    Returns:
        List of synthetic examples for single_point_track_per_frame
    """
    synthetic_examples = []
    
    # Extract dataset parameters from task name
    sampling_fps = -1
    interval_seconds = -1
    
    if "sample_fps_" in base_task_name:
        sample_fps_match = re.search(r'sample_fps_([\d.]+)', base_task_name)
        if sample_fps_match:
            sampling_fps = float(sample_fps_match.group(1))
    elif "interval_seconds_" in base_task_name:
        interval_match = re.search(r'interval_seconds_([\d.]+)', base_task_name)
        if interval_match:
            interval_seconds = float(interval_match.group(1))
    
    # Get original examples metadata
    if isinstance(ground_start_end_data, list):
        original_examples = {ex.get('example_id', ex.get('id')): ex for ex in ground_start_end_data}
    else:
        original_examples = {ex.get('example_id', ex.get('id')): ex 
                           for ex in ground_start_end_data.get('predictions', [])}
    
    # Create synthetic examples for each grounded object
    for example_id, objects in grounded_objects.items():
        original_ex = original_examples[example_id]
        w,h = original_ex['w'], original_ex['h']
        expression = original_ex['expression']
        
        for obj_idx, obj in enumerate(objects):
            # Create single_point_track_per_frame prompt
            start_point = obj['start_point']
            start_point_pixel = get_absolute_coordinates(start_point, w, h)
            start_time = obj['start_time']
            end_time = obj['end_time']

            # Generate prompt for single_point_track_per_frame
            prompt = PromptGenerator.get_prompts(
                'single_point_track_per_frame',
                expr=expression,
                start_point=start_point,
                start_time=start_time,
                sampling_fps=sampling_fps,
                interval_seconds=interval_seconds
            )[0]
            
            synthetic_example = {
                'id': f"{example_id}_{obj_idx}",
                'video': original_ex['video'],
                'prompt': prompt,
                'prompt_type': 'single_point_track_per_frame',
                'expression': expression,
                'start_point': start_point,
                'start_point_pixel': start_point_pixel,
                'start_time': start_time,
                'end_time': end_time,
                'object_index': obj_idx,
                'metadata': {
                    'w': w,
                    'h': h,
                    'mask_id': ['0']
                }
            }
            
            synthetic_examples.append(synthetic_example)
    
    return synthetic_examples


def aggregate_pointing_metrics(
    ground_start_end_results_path: str,
    single_point_track_results_path: str,
    multi_tracking_dataset: ObjectTracking,
    enforce_grounding_end: bool=True,
) -> Dict[str, Any]:
    """
    Aggregate metrics from both stages of pointing evaluation.
    
    Args:
        ground_start_end_results_path: Path to point_ground_start_end predictions
        single_point_track_results_path: Path to single_point_track_per_frame predictions
        multi_tracking_dataset: Video pointing dataset containing ground truth masks
        enforce_grounding_end: If True, use grounding end points as last known points in tracking
        
    Returns:
        Combined metrics including two-stage F1 score
    """
    metrics = {}
    
    # Load predictions from both stages
    with open(ground_start_end_results_path, 'r') as f:
        ground_start_end_data = json.load(f)
    
    with open(single_point_track_results_path, 'r') as f:
        single_point_data = json.load(f)
    
    # Extract metrics if available
    if isinstance(ground_start_end_data, dict) and 'metrics' in ground_start_end_data:
        metrics['point_ground_start_end_metrics'] = ground_start_end_data['metrics']
    
    if isinstance(single_point_data, dict) and 'metrics' in single_point_data:
        metrics['single_point_track_metrics'] = single_point_data['metrics']
    
    # Count statistics
    n_videos_grounded = len(set(ex.get('example_id', ex.get('id', '')) 
                               for ex in ground_start_end_data 
                               if isinstance(ground_start_end_data, list)))
    
    n_objects_tracked = len(single_point_data) if isinstance(single_point_data, list) else \
                       len(single_point_data.get('predictions', []))
    
    metrics['statistics'] = {
        'n_videos_grounded': n_videos_grounded,
        'n_objects_tracked': n_objects_tracked,
        'avg_objects_per_video': n_objects_tracked / n_videos_grounded if n_videos_grounded > 0 else 0
    }
    
    # TODO: Compute actual F1 scores when evaluation metrics are available
    # This would require loading the original HF dataset with masks and computing point-in-mask accuracy

    # Gather start and end points from grounding stage
    grounded_objects = parse_grounding_predictions(ground_start_end_data)

    # Gather single point tracking per query_id
    single_point_results = defaultdict(list)
    for data in single_point_data:
        single_point_prediction = data['prediction']
        qid, obj_id = data['example_id'].rsplit('_', 1)
        single_point_trajectory = parse_single_point_predictions_to_point_trajectory(
            single_point_prediction, 
            video_width=data['w'], video_height=data['h'], video_fps=data['video_fps']
        )

        # Add start, end points from grounding stage if available
        grounded_obj = grounded_objects[qid][int(obj_id)]
        # Add start point
        single_point_trajectory.insert(0,{
            'frame': int(grounded_obj['start_time'] * grounded_obj['video_fps']),
            'time': format_time(grounded_obj['start_time']),
            'points': {0: {'point': get_absolute_coordinates(grounded_obj['start_point'], data['w'], data['h'])}}
        })

        if enforce_grounding_end:
            end_time: float = grounded_obj['end_time']
            single_point_trajectory = [p for p in single_point_trajectory if p['frame'] <= int(end_time * grounded_obj['video_fps'])]
        single_point_results[qid].append(single_point_trajectory)
    
    # Run eval in multi-object tracking
    failed_videos = []
    video_results = []
    for qid, single_point_trajectories in single_point_results.items():

        try:

            # Get predicted trajectory per frame, merging multiple objects
            pred_trajectory_per_frame = {}
            for obj_id, trajs in enumerate(single_point_trajectories):
                for traj in trajs:
                    frame_idx = traj['frame']
                    if frame_idx not in pred_trajectory_per_frame:
                        pred_trajectory_per_frame[frame_idx] = {
                            'frame': frame_idx,
                            'time': traj['time'],
                            'points': {}
                        }
                    pred_trajectory_per_frame[frame_idx]['points'][obj_id] = traj['points'][0]
            pred_trajectory = sorted(list(pred_trajectory_per_frame.values()), key=lambda x: x['frame'])

            # Gather gt annotations for computing metric
            metadata = multi_tracking_dataset.get_by_example_id(qid)['metadata']
            gt_trajectory = metadata['points']
            video_metrics = evaluate_video_point_tracking_with_masks(
                pred_trajectory,
                gt_trajectory,
                metadata
            )

            video_metrics['example_id'] = qid
            video_results.append(video_metrics)
        
        except Exception as e:
            print(f"Failed to evaluate video {qid}: {e}")
            failed_videos.append(qid)
            continue
    
    # Compute overall statistics
    if video_results:
        overall_precision = np.mean([r['precision'] for r in video_results])
        overall_recall = np.mean([r['recall'] for r in video_results])
        overall_f1 = np.mean([r['f1'] for r in video_results])
        total_frames = sum([r['num_frames'] for r in video_results])
        
        # Compute diagnostic statistics
        diagnostic_stats = {
            'total_frames_with_pred': sum([r['frames_with_pred'] for r in video_results]),
            'total_frames_with_gt': sum([r['frames_with_gt'] for r in video_results]),
            'total_frames_with_both': sum([r['frames_with_both'] for r in video_results]),
            'total_frames_pred_only': sum([r['frames_pred_only'] for r in video_results]),
            'total_frames_gt_only': sum([r['frames_gt_only'] for r in video_results])
        }
    else:
        overall_precision = overall_recall = overall_f1 = 0.0
        total_frames = 0
        diagnostic_stats = {
            'total_frames_with_pred': 0,
            'total_frames_with_gt': 0,
            'total_frames_with_both': 0,
            'total_frames_pred_only': 0,
            'total_frames_gt_only': 0
        }
    
    results = {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'num_videos_evaluated': len(video_results),
        'num_videos_failed': len(failed_videos),
        'total_frames_evaluated': total_frames,
        'diagnostic_stats': diagnostic_stats,
        'video_results': video_results,
        'failed_videos': failed_videos
    }
    
    return results