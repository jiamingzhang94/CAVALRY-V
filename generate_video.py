import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

import json
import config

# from baseline import *
from model_utils import Generator

parser = argparse.ArgumentParser(description='Enhanced Video Adversarial Attack')
parser.add_argument('--use_baseline', type=str, default='our')
parser.add_argument('--model_path', type=str, default=config.DEFAULT_MODEL, help='Model path')
parser.add_argument('--qa_json', type=str, default='dataset/mmbench-video-qa.json', help='Path to QA JSON file')
parser.add_argument('--epsilon', type=float, default=0.0627, help='Maximum perturbation size (L∞ norm)')
parser.add_argument('--print_frequency', type=int, default=10, help='Print progress frequency')
parser.add_argument('--token', type=str, default=None, help='HuggingFace token for model download')
parser.add_argument('--batch_processing', action='store_true', default=True, help='Whether to batch process all videos in JSON')
parser.add_argument('--max_videos', type=int, default=0, help='Maximum number of videos to process (0 for all)')
parser.add_argument('--skip_existing', action='store_true', default=True, help='Whether to skip already processed videos')
parser.add_argument('--bs', type=int, default=300, help='Batch size for frame processing')
parser.add_argument('--video_input_dir', type=str, default='dataset/mmbench-video/video', help='Video input directory')
parser.add_argument('--video_output_dir', type=str, default='dataset/mmbench-video-adv/video', help='Video output directory')
parser.add_argument('--generator_checkpoint', type=str, default='./weights/video/finetune/checkpoint_latest.pth', help='Path to generator checkpoint')

args = parser.parse_args()

device = torch.device("cuda")

os.makedirs(args.video_output_dir, exist_ok=True)

def create_adversarial_video(video_path, generator, epsilon=0.1, output_path=None, device="cuda"):
    import os
    from decord import VideoReader
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    height, width = vr[0].shape[0:2]
    total_frames = len(vr)
    print(f"Original video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    model_size = config.IMAGE_SIZE if isinstance(config.IMAGE_SIZE, int) else config.IMAGE_SIZE[0]
    
    batch_size = args.bs
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        use_imageio = True
        print(f"Using imageio to write video")
    except Exception as e:
        print(f"imageio import error: {e}, using OpenCV fallback")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        use_imageio = False
    
    print(f"Starting to process {total_frames} frames...")
    for i in range(num_batches):
        print(f"Processing batch {i+1}/{num_batches}")
        
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_frames)
        batch_indices = list(range(start_idx, end_idx))
        
        batch_frames = vr.get_batch(batch_indices).asnumpy()
        
        original_frames = torch.from_numpy(batch_frames).float().permute(0, 3, 1, 2) / 255.0
        original_frames = original_frames.to(device)
        
        original_size = (height, width)
        
        resized_frames = F.interpolate(
            original_frames, 
            size=(model_size, model_size) if isinstance(model_size, int) else model_size,
            mode='bilinear', 
            align_corners=False
        )
        
        with torch.no_grad():

            noise = generator(resized_frames)
            
            noise = torch.clamp(noise, -args.epsilon, args.epsilon)
            noise = F.interpolate(
                noise, 
                size=original_size,
                mode='bilinear', 
                align_corners=False
            )
            
            perturbed_frames = original_frames + noise
            perturbed_frames = torch.clamp(perturbed_frames, 0.0, 1.0)
        
        perturbed_frames = perturbed_frames.permute(0, 2, 3, 1)
        perturbed_frames = (perturbed_frames * 255.0).to(torch.uint8).cpu().numpy()
        
        for j in range(perturbed_frames.shape[0]):
            if use_imageio:
                writer.append_data(perturbed_frames[j])
            else:
                frame_bgr = cv2.cvtColor(perturbed_frames[j], cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        print(f"Processed {end_idx}/{total_frames} frames ({(end_idx/total_frames)*100:.1f}%)")
    
    if use_imageio:
        writer.close()
    else:
        out.release()
    
    print(f"Successfully saved adversarial video to: {output_path}")
    return output_path

def load_qa_data(json_path, video_id=None):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        import random
        if isinstance(qa_data, dict):
            keys = list(qa_data.keys())
            random.shuffle(keys)
            qa_data = {k: qa_data[k] for k in keys}
        elif isinstance(qa_data, list):
            random.shuffle(qa_data)
        
        if video_id is not None:
            if video_id in qa_data:
                return qa_data[video_id]
            else:
                print(f"Warning: Failed to find video ID {video_id}")
                return []
        else:
            return qa_data
    except Exception as e:
        print(f"Failed to load json: {e}")
        return {}


def batch_process_videos(generator):
    all_qa_data = load_qa_data(args.qa_json)
    video_ids = list(all_qa_data.keys())
    local_video_ids = video_ids

    for i, video_id in enumerate(local_video_ids):
        video_path = os.path.join(args.video_input_dir, f"{video_id}")
        output_path = os.path.join(args.video_output_dir, f"{video_id}")

        if args.skip_existing and os.path.exists(output_path):
            print(f"Skipping existing output video: {output_path}")
            continue
        if not os.path.exists(video_path):
            print(f"Error: Video file not found, skipping: {video_path}")
            continue

        placeholder_path = f"{output_path}.inprogress"
        process_id = os.getpid()
        
        if os.path.exists(placeholder_path):
            print(f"Warning: Video {video_id} is already being processed, skipping")
            continue

        with open(placeholder_path, 'w') as f:
            f.write(f"PID:{process_id}")
        print(f"Created placeholder file: {placeholder_path} indicating video {video_id} is being processed")

        create_adversarial_video(video_path=video_path,
                                 generator=generator,
                                 epsilon=args.epsilon,
                                 output_path=output_path,
                                 device=device)
        
        if os.path.exists(placeholder_path):
            os.remove(placeholder_path)
            print(f"Removed placeholder file: {placeholder_path}")

if __name__ == "__main__":
    os.makedirs(args.video_output_dir, exist_ok=True)
    if 'our' in args.use_baseline:
        generator = Generator().to(device).to(torch.float32)
        generator_checkpoint = args.generator_checkpoint
        checkpoint = torch.load(generator_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['state_dict'])
        generator = generator.eval()
        print(f"Loaded generator weights from {generator_checkpoint}")
    else:
        generator = None

    batch_process_videos(generator=generator)
