import functools
import os
import argparse
import logging
from collections import defaultdict
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from PIL import Image, ImageFile, ImageDraw
import PIL
## Turn off multiprocessing to make the scheduling deterministic
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_VIDEO_LOADER_BACKEND"] = "molmo2"

import numpy as np
import torch

from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.util import (
    prepare_cli_environment,
    resource_path,
)
from olmo.html_utils import postprocess_prompt

import gradio as gr

from vllm import LLM
from vllm.sampling_params import SamplingParams

from transformers import AutoProcessor

try:
    from molmo_utils import process_vision_info
except ImportError:
    # raise ImportError("molmo_utils not found. Please install it with `pip install molmo-utils`.")
    pass


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


CURRENT_MODEL: str = ""
CACHE = "model_cache"
log = logging.getLogger(__name__)
ALLOWED_PATH = [CACHE]
MAX_IMAGE_SIZE = 512
MAX_VIDEO_HEIGHT = 512
POINT_SIZE = 0.01


def draw_points(image, points):
    if isinstance(image, np.ndarray):
        annotation = PIL.Image.fromarray(image)
    else:
        annotation = image.copy()
    draw = ImageDraw.Draw(annotation)
    w, h = annotation.size
    size = max(5, int(max(w, h) * POINT_SIZE))
    for x, y in points:
        draw.ellipse((x-size, y-size, x+size, y+size), fill="rgb(240, 82, 156)", outline=None)
    return annotation


def get_message(
    images: list[Image.Image] | None,
    video_path: str | None,
    max_frames: int,
    frame_sample_mode: str,
    max_fps: int | None,
    sampling_fps: int | None,
    input_text: str,
    style: str,
):
    content = [
        dict(type="text", text=input_text, stye=style)
    ]
    if images:
        image_content = [
            dict(type="image", image=image)
            for image in images
        ]
        content.extend(image_content)
    if video_path:
        video_kwargs = {
            "max_frames": max_frames,
            "frame_sample_mode": frame_sample_mode,
        }
        if max_fps is not None:
            video_kwargs["max_fps"] = max_fps
        if sampling_fps is not None:
            video_kwargs["sampling_fps"] = sampling_fps
        video_content = dict(type="video", video=video_path, **video_kwargs)
        content.append(video_content)

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def run_single_inference(*inputs, annotations=None):
    video_path, images, input_text, style, frame_sample_mode, max_frames, max_fps, sampling_fps, max_steps = inputs
    assert images is not None or video_path is not None, "Either images or video_path must be provided"
    assert images is None or video_path is None, "Both images and video_path cannot be provided at the same time"
    logging.info(f"Running inference for question: \"{input_text}\", style={style} steps={max_steps}")
    nimages = 0
    if images:
        images = [t[0] for t in images]
        nimages = len(images)
        logging.info(f"# of images: {nimages}")

    messages = get_message(
        images=images,
        video_path=video_path,
        max_frames=max_frames,
        frame_sample_mode=frame_sample_mode,
        max_fps=max_fps,
        sampling_fps=sampling_fps,
        input_text=input_text,
        style=style,
    )
    images, videos_inputs, video_kwargs = process_vision_info(messages)
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos_inputs:
        videos, video_metadatas = zip(*videos_inputs)
        videos, video_metadatas = list(videos), list(video_metadatas)
        logging.info(
            f"Videos: {videos[0].shape}, frame_sample_mode: {frame_sample_mode}, "
            f"max_frames: {max_frames}, max_fps: {max_fps}, sampling_fps: {sampling_fps}"
        )
        multi_modal_data["video"] = videos_inputs
    else:
        videos = None
        video_metadatas = None
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vllm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": video_kwargs,
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=max_steps,
        temperature=0
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    prompt_text = postprocess_prompt(processor.decode(outputs[0].prompt_token_ids))
    logging.info(f"vllm prompt: {prompt_text}")
    logging.info(f"vllm generated_text: {generated_text}")
    if annotations:
        if video_path is None and nimages == 1:
            w, h = images[0].size
            points = point_formatter.extract_points(generated_text, w, h)
            if points:
                return generated_text, [draw_points(images[0], points)]
            else:
                return generated_text, []
        elif video_path is None and nimages > 1:
            w, h = [x.size[0] for x in images], [x.size[1] for x in images]
            points = point_formatter.extract_multi_image_points(generated_text, w, h)
            if points:
                group_by_index = defaultdict(list)
                for ix, x, y in points:
                    group_by_index[ix].append((x, y))
                out = []
                for ix, points in group_by_index.items():
                    out.append(draw_points(images[ix-1], points))
                return generated_text, out
            else:
                return generated_text, []
        else:
            h, w = videos[0].shape[1:3]
            group_by_time = defaultdict(list)
            points = point_formatter.extract_multi_image_points(generated_text, w, h)
            if points:
                for ts, x, y in points:
                    group_by_time[ts].append((x, y))
            else:
                track = point_formatter.extract_trajectories(generated_text, w, h, 30)
                for ex in track:
                    group_by_time[ex["time"]] = [(x["x"], x["y"]) for x in ex["points"]]
            grouped_by_frame = defaultdict(list)
            for ts, points in group_by_time.items():
                timestamps = video_metadatas[0]["frames_indices"] / video_metadatas[0]["fps"]
                ix = int(np.argmin(np.abs(timestamps - ts)))
                grouped_by_frame[ix] += points
            out = []
            for ix, points in grouped_by_frame.items():
                out.append(draw_points(videos[0][ix], points))
            return generated_text, out
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Location of Molmo2 checkpoint.")
    parser.add_argument("--server_name")
    parser.add_argument("--use_float32", action="store_true", help="Use float32 weights")
    parser.add_argument("--default_max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="Fraction of GPU memory to use.")
    parser.add_argument("--original_ckpt_home", type=str, default=None)
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--cloudflare_tunnel", action="store_true")
    parser.add_argument("--no_share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    prepare_cli_environment()

    model_dir = args.model_dir
    if model_dir.startswith("gs://"):
        if model_dir.endswith("/"):
            model_dir = model_dir[:-1]
        model_cache_dir = os.path.join(os.environ.get("MOLMO_CACHE_DIR"), "ckpt")
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model files from {model_dir} to {model_cache_dir}...")
        try:
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_dir}/*", f"{model_cache_dir}/"], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download model files from {model_dir} to {model_cache_dir}: {e}")
        model_dir = model_cache_dir

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float32" if args.use_float32 else "bfloat16",
        limit_mm_per_prompt={"image": 6, "video": 1},
        max_num_batched_tokens=36864,
    )

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        padding_side="left",
    )

    if args.annotations:
        assert args.original_ckpt_home is not None, "original_ckpt_home must be provided when annotations are enabled"
        model_cfg_path = resource_path(args.original_ckpt_home, "config.yaml")
        model_cfg = Molmo2Config.load(model_cfg_path, key="model", validate_paths=False)
        preprocessor = model_cfg.build_preprocessor(for_inference=True, is_training=False)
        point_formatter = preprocessor.formatter._point_formatter

    CSS = """
    #input_image image {
        object-fit: contain !important;
    }
    #input_video video {
        object-fit: contain !important;
    }
    """

    frame_sample_mode = processor.video_processor.frame_sample_mode
    max_frames = processor.video_processor.num_frames
    max_fps = processor.video_processor.max_fps
    sampling_fps = processor.video_processor.sampling_fps

    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            ## Molmo2 Demo
            Provide either a video or images and a prompt for question answering.
            """
        )

        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("video"):
                    video = gr.Video(label="Input Video", elem_id="input_video", height=MAX_VIDEO_HEIGHT)
                with gr.TabItem("image(s)"):
                    images = gr.Gallery(label="Input Images", elem_id="input_image", type="pil", height=MAX_IMAGE_SIZE)

        with gr.Row():
            input_text = gr.Textbox(placeholder="Enter the prompt", label="Input text")

        with gr.Row():
            style = gr.Textbox(value="demo", label="style")
            frame_sample_mode = gr.Textbox(value=frame_sample_mode, label="frame_sample_mode")
            max_frames = gr.Number(value=max_frames, label="max_frames")
            max_fps = gr.Number(value=max_fps, label="max_fps")
            sampling_fps = gr.Number(value=sampling_fps, label="sampling_fps")
            max_tok_slider = gr.Slider(label="max_tokens", minimum=1, maximum=4096, step=1, value=args.default_max_tokens)

        with gr.Row():
            submit_button = gr.Button("Submit", scale=3)
            clear_all_button = gr.ClearButton(components=[video, images, input_text], value="Clear All", scale=1)

        with gr.Row():
            output_text = gr.Textbox(placeholder="Output text", label="Output text", lines=10)

        if args.annotations:
            with gr.Row():
                output_annotations = gr.Gallery(label="Annotations", height=MAX_IMAGE_SIZE)
            outputs = [output_text, output_annotations]
            fn = functools.partial(run_single_inference, annotations="points")
        else:
            fn = run_single_inference
            outputs = [output_text]

        submit_button.click(
            fn=fn,
            inputs=[video, images, input_text, style, frame_sample_mode, max_frames, max_fps, sampling_fps, max_tok_slider],
            outputs=outputs,
        )

    if args.cloudflare_tunnel:
        import cloudflared_tunnel
        with cloudflared_tunnel.run() as port:
            demo.queue().launch(
                share=False, show_error=True, max_threads=os.cpu_count() - 10, server_port=port,
                allowed_paths=ALLOWED_PATH, css=CSS,
            )
    else:
        demo.queue().launch(
            server_name=args.server_name,
            share=not args.no_share, show_error=True, max_threads=os.cpu_count() - 10,
            server_port=args.port,
            allowed_paths=ALLOWED_PATH,
            css=CSS,
        )
