#!/usr/bin/env python3
"""
Ego4D clip extraction script from custom JSONL file with sharding support.

This script extracts video clips from a custom JSONL file containing video paths
and start/end times. Each row in the JSONL should have:
- video_path: Path to the video file
- start_time: Start time in seconds (float)
- end_time: End time in seconds (float)
- Any additional metadata fields (optional)

Usage:
    python extract_ego4d_clips.py --shard_id 0 --num_shards 4 --clips_jsonl /path/to/ego4d_clips.jsonl
"""

import argparse
import os
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import ffmpeg
from tqdm import tqdm
import hashlib
import json
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths - adjust these according to your setup.
DEFAULT_CLIPS_JSONL = os.environ.get("EGO4D_CLIPS_JSONL", "ego4d_clips.jsonl")
DEFAULT_OUTPUT_PATH = os.environ.get("EGO4D_CLIPS_OUTPUT", "ego4d-clips")


def get_video_hash(video_path):
    """Get a hash of the video path for consistent sharding."""
    return hashlib.md5(video_path.encode()).hexdigest()


def extract_clip(video_path, start_time, end_time, output_path):
    """
    Extract a clip from a video using ffmpeg.

    Args:
        video_path (str): Path to the input video
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_path (str): Path for the output clip

    Returns:
        str: Path to the extracted clip, or None if extraction failed
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if clip already exists
        if os.path.exists(output_path):
            logger.debug(f"Clip already exists: {output_path}")
            return output_path

        # Calculate duration
        duration = end_time - start_time
        if duration <= 0:
            logger.warning(f"Invalid duration {duration:.3f}s for clip: {output_path}")
            return None

        # Extract clip using ffmpeg with error capture
        input_stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            vcodec='libx264',
            acodec='aac',
            loglevel='warning'
        )

        # Run ffmpeg and capture stdout/stderr
        try:
            ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            # Log the actual ffmpeg error output
            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
            stdout_output = e.stdout.decode('utf-8') if e.stdout else "No stdout output"
            logger.error(f"FFmpeg error for {output_path}:")
            logger.error(f"Command: {' '.join(e.cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"STDERR: {stderr_output}")
            logger.error(f"STDOUT: {stdout_output}")
            return None

        # Verify the output file was created and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                logger.debug(f"Successfully extracted clip: {output_path} ({file_size} bytes)")
                return output_path
            else:
                logger.error(f"Extracted clip has zero size: {output_path}")
                # Remove the empty file
                os.remove(output_path)
                return None
        else:
            logger.error(f"Output file was not created: {output_path}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error extracting clip {output_path}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def get_video_hash(video_path):
    """Get a hash of the video path for consistent sharding."""
    return hashlib.md5(video_path.encode()).hexdigest()


def process_video_clips(video_data, output_path, max_workers=4):
    """
    Process all clips for a single video in parallel.

    Args:
        video_data (tuple): Tuple of (video_path, clips_list)
        output_path (str): Base output path
        max_workers (int): Number of worker threads

    Returns:
        list: List of successfully extracted clips with metadata
    """
    video_path, clips = video_data

    # Check if video file exists
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return []

    # Additional validation: check if file is readable and has reasonable size
    try:
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.warning(f"Video file has zero size: {video_path}")
            return []
        logger.debug(f"Processing video: {video_path} ({file_size} bytes, {len(clips)} clips)")
    except OSError as e:
        logger.error(f"Cannot access video file {video_path}: {str(e)}")
        return []

    extracted_clips = []

    def extract_single_clip(clip_info):
        start_time, end_time, task_type, metadata, row_index = clip_info

        # Validate clip times
        if start_time >= end_time:
            logger.warning(f"Invalid clip times: start={start_time}, end={end_time} for {video_path}")
            return None

        duration = end_time - start_time
        if duration < 0.1:  # Very short clips might be problematic
            logger.warning(f"Very short clip ({duration:.3f}s): {video_path} [{start_time}-{end_time}]")

        # Create output filename
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        clip_filename = f"{video_id}_{start_time:.3f}_{end_time:.3f}_{task_type}.mp4"
        clip_output_path = os.path.join(output_path, clip_filename)

        # Extract the clip
        result_path = extract_clip(video_path, start_time, end_time, clip_output_path)

        if result_path:
            return {
                'video_id': video_id,
                'original_video': video_path,
                'clip_path': result_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'task_type': task_type,
                'metadata': metadata,
                'row_index': row_index
            }
        return None

    # Process clips for this video in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {
            executor.submit(extract_single_clip, clip_info): clip_info
            for clip_info in clips
        }

        for future in as_completed(future_to_clip):
            result = future.result()
            if result:
                extracted_clips.append(result)

    return extracted_clips


def merge_overlapping_moments(moments, new_start, new_end, threshold=0.75):
    """Helper function to check overlap and merge moments (from Ego4d class)."""
    for i, (existing_start, existing_end) in enumerate(moments):
        intersection_start = max(new_start, existing_start)
        intersection_end = min(new_end, existing_end)
        intersection = max(0, intersection_end - intersection_start)

        new_len = new_end - new_start
        existing_len = existing_end - existing_start

        if new_len > 0 and existing_len > 0:
            new_overlap_ratio = intersection / new_len
            existing_overlap_ratio = intersection / existing_len

            if new_overlap_ratio > threshold or existing_overlap_ratio > threshold:
                moments.pop(i)
                if intersection_end > intersection_start:
                    moments.append((intersection_start, intersection_end))
                return True
    return False


def load_and_shard_data(clips_jsonl_path, shard_id, num_shards):
    """
    Load Ego4D clips metadata from JSONL file and return data for the specified shard.

    Args:
        clips_jsonl_path (str): Path to the JSONL file containing video paths and clip times
        shard_id (int): Current shard ID (0-indexed)
        num_shards (int): Total number of shards

    Returns:
        dict: Dictionary mapping video paths to list of clip information
    """
    if not os.path.exists(clips_jsonl_path):
        raise FileNotFoundError(f"Clips JSONL file not found: {clips_jsonl_path}")

    logger.info(f"Loading clips from: {clips_jsonl_path}")

    # Read JSONL file
    clips_data = []
    with open(clips_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                clips_data.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num + 1}: {e}")
                continue

    logger.info(f"Loaded {len(clips_data)} clips from JSONL file")

    # Group clips by video path
    video_clips = defaultdict(list)
    skipped = 0

    for idx, clip_data in enumerate(clips_data):
        try:
            video_path = clip_data['video']
            start_time = float(clip_data['clip_start_time'])
            end_time = float(clip_data['clip_end_time'])

            # Validate clip times
            if end_time <= start_time:
                skipped += 1
                continue

            # Extract video filename/ID for consistent naming
            video_filename = os.path.basename(video_path)
            video_id = os.path.splitext(video_filename)[0]

            # Store clip info with metadata
            metadata = {
                'original_index': idx,
                'video_path': video_path,
                **{k: v for k, v in clip_data.items() if k not in ['video_path', 'start_time', 'end_time']}
            }

            video_clips[video_path].append((start_time, end_time, "custom_clip", metadata, idx))

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing clip {idx}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        logger.warning(f"Skipped {skipped} clips due to invalid data.")

    # Get unique videos and shard them
    unique_videos = list(video_clips.keys())
    logger.info(f"Total unique videos: {len(unique_videos)}")

    # Shard videos based on hash for consistent assignment
    shard_videos = []
    for video_path in unique_videos:
        video_hash = get_video_hash(video_path)
        video_shard = int(video_hash, 16) % num_shards
        if video_shard == shard_id:
            shard_videos.append(video_path)

    logger.info(f"Videos in shard {shard_id}/{num_shards}: {len(shard_videos)}")

    # Return only the data for this shard
    shard_data = {video_path: video_clips[video_path] for video_path in shard_videos}

    # Calculate total clips in this shard
    total_clips = sum(len(clips) for clips in shard_data.values())
    logger.info(f"Total clips in shard {shard_id}: {total_clips}")

    return shard_data


def main():
    parser = argparse.ArgumentParser(description="Extract Ego4D clips from custom JSONL file with sharding")
    parser.add_argument("--shard_id", type=int, required=True,
                       help="Shard ID (0-indexed)")
    parser.add_argument("--num_shards", type=int, required=True,
                       help="Total number of shards")
    parser.add_argument("--clips_jsonl", type=str, default=DEFAULT_CLIPS_JSONL,
                       help="Path to JSONL file containing video paths and clip times")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                       help="Output path for extracted clips")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Number of worker threads per video")
    parser.add_argument("--video_workers", type=int, default=1,
                       help="Number of videos to process in parallel")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logger.setLevel(numeric_level)

    # Validate arguments
    if args.shard_id >= args.num_shards or args.shard_id < 0:
        raise ValueError(f"Invalid shard_id {args.shard_id} for num_shards {args.num_shards}")

    logger.info(f"Starting Ego4D clip extraction from custom JSONL")
    logger.info(f"Shard: {args.shard_id}/{args.num_shards}")
    logger.info(f"Clips JSONL: {args.clips_jsonl}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Workers per video: {args.max_workers}")
    logger.info(f"Parallel videos: {args.video_workers}")

    # Load and shard data
    shard_data = load_and_shard_data(
        args.clips_jsonl, args.shard_id, args.num_shards
    )

    if not shard_data:
        logger.info("No videos assigned to this shard. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Process videos in parallel
    all_extracted_clips = []
    video_items = list(shard_data.items())

    def process_single_video(video_data):
        return process_video_clips(
            video_data, args.output_path, args.max_workers
        )

    # Process videos with progress bar
    with ThreadPoolExecutor(max_workers=args.video_workers) as executor:
        with tqdm(total=len(video_items), desc="Processing videos") as pbar:
            future_to_video = {
                executor.submit(process_single_video, (video_path, clips)): video_path
                for video_path, clips in video_items
            }

            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    extracted_clips = future.result()
                    all_extracted_clips.extend(extracted_clips)
                    video_name = os.path.basename(video_path)
                    logger.info(f"Processed {len(extracted_clips)} clips from {video_name}")
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {str(e)}")
                finally:
                    pbar.update(1)

    # Save extraction summary
    summary_file = os.path.join(
        args.output_path,
        f"extraction_summary_shard_{args.shard_id}.json"
    )

    summary = {
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'clips_jsonl': args.clips_jsonl,
        'total_videos_processed': len(video_items),
        'total_clips_extracted': len(all_extracted_clips),
        'extracted_clips': all_extracted_clips
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Extraction complete!")
    logger.info(f"Total videos processed: {len(video_items)}")
    logger.info(f"Total clips extracted: {len(all_extracted_clips)}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
