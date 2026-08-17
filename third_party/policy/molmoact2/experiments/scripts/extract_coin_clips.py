#!/usr/bin/env python3
"""Extract COIN video segments from full-length source videos using ffmpeg.

Reads COIN.json annotations, locates source videos in the `videos/` directory,
and extracts each annotated segment into `video_segments/` as a standalone
.mp4 clip using `ffmpeg -ss ... -to ... -c copy` (stream copy, very fast).

Clips are named: {video_id}_{start}_{end}.mp4
(using the raw float timestamps from the annotations, e.g. -0X2mXPy3Mc_2.0_9.0.mp4)

Existing clips are skipped, so the script is safe to re-run (idempotent).

Usage:
    python scripts/extract_coin_clips.py [--coin-dir DIR] [--workers N] [--re-encode]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import exists, join

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_COIN_DIR = os.environ.get("COIN_DATA_DIR", "coin")
VIDEO_EXTENSIONS = [".mkv", ".mp4", ".webm"]


def find_source_video(video_dir: str, video_id: str, extensions_map: dict | None = None) -> str | None:
    """Find the source video file, trying known extensions."""
    if extensions_map and video_id in extensions_map:
        ext = extensions_map[video_id]
        if ext is None:
            return None
        return join(video_dir, f"{video_id}{ext}")

    for ext in VIDEO_EXTENSIONS:
        path = join(video_dir, f"{video_id}{ext}")
        if exists(path):
            return path
    return None


def extract_clip(
    source_path: str,
    output_path: str,
    start: float,
    end: float,
    re_encode: bool = False,
) -> tuple[bool, str]:
    """Extract a clip from source_path[start:end] -> output_path using ffmpeg.

    Returns (success, message).
    """
    if exists(output_path):
        return True, "already exists"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_path = output_path + ".tmp.mp4"

    if re_encode:
        # Re-encode: slower but guarantees frame-accurate cuts and consistent codec
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", source_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-loglevel", "error",
            tmp_path,
        ]
    else:
        # Stream copy: very fast, no quality loss, but cuts may not be frame-exact
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", source_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            tmp_path,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            if exists(tmp_path):
                os.remove(tmp_path)

            # If stream copy failed, retry with re-encode (handles broken timestamps)
            if not re_encode:
                re_encode_cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-to", str(end),
                    "-i", source_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    tmp_path,
                ]
                retry = subprocess.run(re_encode_cmd, capture_output=True, text=True, timeout=300)
                if retry.returncode == 0:
                    os.rename(tmp_path, output_path)
                    return True, "ok (re-encoded)"
                if exists(tmp_path):
                    os.remove(tmp_path)
                return False, f"ffmpeg error (re-encode also failed): {retry.stderr.strip()[:200]}"

            return False, f"ffmpeg error (rc={result.returncode}): {result.stderr.strip()[:200]}"

        # Atomic rename
        os.rename(tmp_path, output_path)
        return True, "ok"

    except subprocess.TimeoutExpired:
        if exists(tmp_path):
            os.remove(tmp_path)
        return False, "ffmpeg timed out (>300s)"
    except Exception as e:
        if exists(tmp_path):
            os.remove(tmp_path)
        return False, str(e)


def _worker(args):
    """Worker function for parallel extraction."""
    source_path, output_path, start, end, re_encode, video_id = args
    ok, msg = extract_clip(source_path, output_path, start, end, re_encode)
    return video_id, start, end, ok, msg


def main():
    parser = argparse.ArgumentParser(description="Extract COIN video segments using ffmpeg")
    parser.add_argument(
        "--coin-dir", default=DEFAULT_COIN_DIR,
        help=f"Path to the COIN dataset directory (default: {DEFAULT_COIN_DIR})",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel ffmpeg processes (default: 8)",
    )
    parser.add_argument(
        "--re-encode", action="store_true",
        help="Re-encode clips (slower but frame-accurate). Default is stream copy (fast).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without actually extracting clips.",
    )
    args = parser.parse_args()

    coin_dir = args.coin_dir
    video_dir = join(coin_dir, "videos")
    segments_dir = join(coin_dir, "video_segments")
    coin_json_path = join(coin_dir, "COIN.json")

    # Validate paths
    if not exists(coin_json_path):
        log.error(f"COIN.json not found at {coin_json_path}")
        sys.exit(1)
    if not exists(video_dir):
        log.error(f"Videos directory not found at {video_dir}")
        sys.exit(1)

    os.makedirs(segments_dir, exist_ok=True)

    # Load annotations
    log.info(f"Loading annotations from {coin_json_path}")
    with open(coin_json_path) as f:
        data = json.load(f)

    # Try to load cached extensions map for faster lookups
    extensions_map = None
    for ext_file in ["file_extensions.json"]:
        ext_path = join(coin_dir, ext_file)
        if exists(ext_path):
            with open(ext_path) as f:
                extensions_map = json.load(f)
            log.info(f"Loaded cached extensions from {ext_file}")
            break

    # Build list of extraction tasks
    tasks = []
    missing_videos = 0
    total_annotations = 0

    for video_id, v in data["database"].items():
        source_path = find_source_video(video_dir, video_id, extensions_map)
        if source_path is None:
            missing_videos += 1
            continue

        for ann in v["annotation"]:
            segment = ann.get("segment")
            if segment is None:
                continue
            start, end = segment
            if end <= start:
                continue

            total_annotations += 1
            clip_name = f"{video_id}_{start}_{end}.mp4"
            output_path = join(segments_dir, clip_name)

            tasks.append((source_path, output_path, start, end, args.re_encode, video_id))

    already_exist = sum(1 for _, out, *_ in tasks if exists(out))
    to_extract = len(tasks) - already_exist

    log.info(f"Total annotated segments: {total_annotations}")
    log.info(f"Missing source videos: {missing_videos}")
    log.info(f"Clips already extracted: {already_exist}")
    log.info(f"Clips to extract: {to_extract}")

    if args.dry_run:
        log.info("Dry run — no clips will be extracted.")
        for source, output, start, end, _, vid in tasks:
            if not exists(output):
                print(f"  {vid} [{start}-{end}] -> {os.path.basename(output)}")
        sys.exit(0)

    if to_extract == 0:
        log.info("Nothing to do — all clips already extracted.")
        sys.exit(0)

    # Extract clips
    log.info(f"Extracting {to_extract} clips using {args.workers} workers...")
    succeeded = 0
    failed = 0
    skipped = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            video_id, start, end, ok, msg = future.result()
            if msg == "already exists":
                skipped += 1
            elif ok:
                succeeded += 1
            else:
                failed += 1
                failures.append((video_id, start, end, msg))

            if i % 500 == 0 or i == len(tasks):
                log.info(
                    f"Progress: {i}/{len(tasks)} "
                    f"(extracted={succeeded}, skipped={skipped}, failed={failed})"
                )

    # Summary
    log.info("=" * 60)
    log.info(f"Extraction complete: {succeeded} extracted, {skipped} skipped, {failed} failed")
    if failures:
        log.warning(f"\n{len(failures)} failed extractions:")
        for vid, s, e, msg in failures[:20]:
            log.warning(f"  {vid} [{s}-{e}]: {msg}")
        if len(failures) > 20:
            log.warning(f"  ... and {len(failures) - 20} more")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
