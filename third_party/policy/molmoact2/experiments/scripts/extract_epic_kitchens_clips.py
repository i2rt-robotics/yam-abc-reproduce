import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def parse_timestamp(timestamp_str):
    """Parse timestamp string like '00:00:01.089' to seconds."""
    parts = timestamp_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def extract_clip(row, videos_root, clips_root):
    participant = row['participant_id']
    video_id = row['video_id']
    start = parse_timestamp(row['start_timestamp'])
    end = parse_timestamp(row['stop_timestamp'])

    start_str = f'{start:.3f}'
    end_str = f'{end:.3f}'

    input_video = os.path.join(videos_root, participant, 'videos', f'{video_id}.MP4')

    # Skip if source video doesn't exist
    if not os.path.exists(input_video):
        return None

    out_dir = os.path.join(clips_root, participant, 'videos')
    os.makedirs(out_dir, exist_ok=True)
    # video_id already includes participant_id (e.g., "P02_104")
    out_clip = os.path.join(out_dir, f'{video_id}_{start_str}_{end_str}.mp4')

    if os.path.exists(out_clip):
        return None

    cmd_copy = [
        'ffmpeg', '-y',
        '-hide_banner', '-loglevel', 'error',
        '-ss', str(start),
        '-to', str(end),
        '-i', input_video,
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        out_clip
    ]
    try:
        subprocess.run(cmd_copy, check=True)
    except subprocess.CalledProcessError:
        print(f'Stream copy failed for {out_clip}, retrying with re-encode...')
        cmd_reencode = [
            'ffmpeg', '-y',
            '-hide_banner', '-loglevel', 'error',
            '-ss', str(start),
            '-to', str(end),
            '-i', input_video,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            out_clip
        ]
        try:
            subprocess.run(cmd_reencode, check=True)
        except subprocess.CalledProcessError:
            print(f'Both stream copy and re-encode failed for {out_clip}')

def main():
    parser = argparse.ArgumentParser(description='Epic Kitchens clip extraction')
    parser.add_argument('--annotation-path', type=str, required=True,
                        help='Path to EPIC_100_train_with_missing.csv')
    parser.add_argument('--videos-root', type=str, required=True,
                        help='Path to original videos root')
    parser.add_argument('--clips-root', type=str, required=True,
                        help='Path to output clips root')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4),
                        help='Number of threads for extraction')
    args = parser.parse_args()

    ann = pd.read_csv(args.annotation_path)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_clip, row, args.videos_root, args.clips_root) for idx, row in ann.iterrows()]
        for future in as_completed(futures):
            future.result()

if __name__ == '__main__':
    main()
