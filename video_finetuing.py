import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import random
import json
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as vision_models
import cv2
import traceback
import logging
import time
import datetime
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader

import model_utils
import attack_utils
from model_utils import Generator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_attack")

def parse_args():
    parser = argparse.ArgumentParser(description='Video Attack')
    parser.add_argument('--model_path', type=str, default='.cache/models--OpenGVLab--InternVL2_5-1B/snapshots/ec696efff1e392b75c46ef1710017ea6244423d1', help='Model path')
    parser.add_argument('--video_path', type=str, default=None, help='Video input path (single video mode)')
    parser.add_argument('--qa_json', type=str, default='dataset/visual.json', help='QA JSON path')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--epsilon', type=float, default=0.06274509, help='Maximum perturbation size (L∞ norm)')

    parser.add_argument('--frame_sample_rate', type=int, default=1)
    parser.add_argument('--keyframe_count', type=int, default=8, help='Number of keyframes to optimize in each iteration')
    parser.add_argument('--frame_gap', type=int, default=30, help='Frame gap between keyframe sets in iteration')
    parser.add_argument('--print_frequency', type=int, default=10, help='Progress printing frequency')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token for model download')
    parser.add_argument('--video_input_dir', type=str, default='/shared/hf-hub/datasets--opencompass--MMBench-Video/snapshots/ef35e21df54488715a906c7e47146f5d9f4abbed/video', help='Video input directory')
    parser.add_argument('--batch_processing', action='store_true', default=True, help='Whether to batch process all videos in JSON')
    parser.add_argument('--max_videos', type=int, default=0, help='Maximum number of videos to process (0 for all)')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Whether to skip already processed videos')
    parser.add_argument('--oom_log_file', type=str, default='oom_videos.json', help='File to log OOM videos')

    parser.add_argument('--semantic_loss', type=float, default=0.1, help='Semantic loss weight')
    parser.add_argument('--feature_loss', type=float, default=20.0, help='Feature loss weight')
    parser.add_argument('--smooth_loss', type=float, default=10.0, help='Smoothness loss weight')

    parser.add_argument('--use_feature_model', action='store_true', default=True, help='Whether to use an additional feature model')
    parser.add_argument('--feature_model', type=str, default='./ckpts/rn50.pth', help='Path to feature model weights (default: pretrained ResNet50)')
    
    parser.add_argument('--initial_lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--final_lr', type=float, default=0.000001, help='Final learning rate')
    
    parser.add_argument('--save_every_n_videos', type=int, default=50, help='Save model every N videos processed')
    parser.add_argument('--dist-url', '--dist_url', default='env://', type=str, help='URL for distributed training')
    parser.add_argument('--dist-backend', '--dist_backend', default='nccl', type=str, help='Backend for distributed training')
    parser.add_argument('--world-size', '--world_size', default=-1, type=int, help='Number of processes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='Global rank of distributed training process')
    parser.add_argument('--gpu', default=None, type=int, help='Specify GPU ID to use')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, help='Rank of process within node')
    parser.add_argument('--multiprocessing-distributed', '--multiprocessing_distributed', action='store_true',
                        help='Use multi-process distributed training, suitable for multi-node training')
    parser.add_argument('--samples_per_gpu', type=int, default=50, 
                       help='Number of samples processed per GPU (corresponds to original iterations)')

    parser.add_argument('--save_path', type=str, default='./weights/', help='Path to save generator weights')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    
    args = parser.parse_args()
    return args

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    
    os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "300"
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.local_rank != -1:
        args.gpu = args.local_rank
        args.rank = args.local_rank
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.gpu = 0 if torch.cuda.is_available() else None
        return

    args.distributed = True

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=300)
    )
    
    try:
        torch.distributed.barrier()
    except Exception as e:
        logger.warning(f"Barrier synchronization failed, but continuing: {str(e)}")
    
    setup_for_distributed(args.rank == 0)


class VideoFramesDataset(Dataset):
    
    def __init__(self, video_path, qa_pairs, frame_sample_rate=1, max_frames=5000, image_size=None):
        self.video_path = video_path
        self.qa_pairs = qa_pairs
        self.image_size = image_size
        
        self.frames, self.frame_indices = self._load_frames(frame_sample_rate, max_frames)
        logger.info(f"Loaded {len(self.frames)} frames from {os.path.basename(video_path)}")
    
    def _load_frames(self, sample_rate, max_frames):
        logger.info(f"Loading video: {self.video_path}")
        
        video_reader = VideoReader(self.video_path, num_threads=4)
        total_frames = len(video_reader)
        
        if max_frames == 0:
            max_frames = total_frames
        
        if sample_rate > 1:
            frame_indices = np.arange(0, total_frames, sample_rate)
        else:
            interval = max(1, total_frames // max_frames)
            frame_indices = np.arange(0, total_frames, interval)
        
        if max_frames > 0 and len(frame_indices) > max_frames:
            indices_subset = np.linspace(0, len(frame_indices)-1, max_frames, dtype=int)
            frame_indices = frame_indices[indices_subset]
        
        logger.info(f"Selected frame indices: {frame_indices[:5]}...{frame_indices[-5:]} (total: {len(frame_indices)})")
        
        frames = video_reader.get_batch(frame_indices).asnumpy()
        
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        if frames_tensor.dtype == torch.uint8:
            frames_tensor = frames_tensor.float() / 255.0
        
        if self.image_size:
            size = self.image_size if isinstance(self.image_size, tuple) else (self.image_size, self.image_size)
            frames_tensor = F.interpolate(
                frames_tensor, 
                size=size,
                mode='bilinear', 
                align_corners=False
            )
        
        return frames_tensor, frame_indices
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx], self.qa_pairs
    
    def get_adaptive_keyframes(self, num_keyframes, total_iterations, current_iteration):
        total_frames = len(self.frames)
        
        if total_frames <= num_keyframes:
            return self.frames
        
        base_stride = total_frames / num_keyframes
        
        if total_iterations > 1:
            offset = (current_iteration * base_stride / total_iterations) % 1.0
        else:
            offset = 0
            
        indices = [int(min(total_frames - 1, (i + offset) * base_stride)) for i in range(num_keyframes)]
        
        indices = sorted(list(set(indices)))
        
        while len(indices) < num_keyframes:
            max_gap = 0
            insert_pos = 0
            for i in range(len(indices) - 1):
                gap = indices[i+1] - indices[i]
                if gap > max_gap:
                    max_gap = gap
                    insert_pos = i
            
            new_idx = indices[insert_pos] + max_gap // 2
            indices.insert(insert_pos + 1, new_idx)
        
        keyframes = torch.stack([self.frames[i] for i in indices])
        
        return keyframes


def log_oom_video(video_id, args):
    if not args.distributed or args.rank == 0:
        oom_log_path = os.path.join(args.output_dir, args.oom_log_file)
        
        oom_videos = {}
        if os.path.exists(oom_log_path):
            try:
                with open(oom_log_path, 'r') as f:
                    oom_videos = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse OOM log file, creating a new one")
                oom_videos = {"oom_videos": []}
        else:
            oom_videos = {"oom_videos": []}
        
        if "oom_videos" not in oom_videos:
            oom_videos["oom_videos"] = []
        
        if video_id not in oom_videos["oom_videos"]:
            oom_videos["oom_videos"].append(video_id)
            
        with open(oom_log_path, 'w') as f:
            json.dump(oom_videos, f, indent=2)
        
        logger.info(f"Video {video_id} logged to OOM log: {oom_log_path}")

def is_oom_error(e):
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
        
    error_msg = str(e).lower()
    oom_indicators = [
        "out of memory",
        "cuda error",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "memory allocation failed",
        "illegal memory access",
        "segmentation fault",
        "device-side assert",
    ]
    
    return any(indicator in error_msg for indicator in oom_indicators)

def train_on_video(video_path, video_id, qa_pairs, tokenizer, model, feature_extractor, 
                  device, args, generator, optimizer):
    try:
        logger.info(f"Processing video: {video_id} ({video_path})")
        
        try:
            dataset = VideoFramesDataset(
                video_path, 
                qa_pairs, 
                frame_sample_rate=args.frame_sample_rate,
                max_frames=5000,
                image_size=448
            )
            
            if len(dataset) == 0:
                logger.error(f"Error: Could not load frames from video {video_id}.")
                return None
                
            logger.info(f"Loaded and preprocessed {len(dataset)} frames")
            
        except Exception as e:
            logger.error(f"Error creating dataset for video {video_id}: {str(e)}")
            return None
        
        total_loss = 0
        for i in range(args.samples_per_gpu):
            try:
                key_frames = dataset.get_adaptive_keyframes(
                    args.keyframe_count, 
                    args.samples_per_gpu, 
                    i
                )
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    key_frames = key_frames.to(device).to(torch.bfloat16).contiguous()
                    
                    noise = generator(key_frames)
                    noise = torch.clamp(noise, -args.epsilon, args.epsilon)
                    
                    adv_frames = key_frames + noise
                    adv_frames = torch.clamp(adv_frames, 0, 1)
                    
                    orig_vlm_features = model.extract_feature(key_frames)
                    adv_vlm_features = model.extract_feature(adv_frames)
                    
                    loss = attack_utils.combined_adversarial_loss(
                        orig_vlm_features, adv_vlm_features,
                        key_frames, adv_frames,
                        qa_pairs, tokenizer, model, feature_extractor, device, args
                    )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                if is_oom_error(e):
                    logger.warning(f"OOM error during iteration {i} for video {video_id}, skipping video")
                    log_oom_video(video_id, args)
                    torch.cuda.empty_cache()
                    return None
                else:
                    logger.error(f"Error during iteration {i} for video {video_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        
        return total_loss / args.samples_per_gpu if args.samples_per_gpu > 0 else 0
        
    except Exception as e:
        if is_oom_error(e):
            logger.warning(f"OOM error while processing video {video_id}: {str(e)}")
            log_oom_video(video_id, args)
            torch.cuda.empty_cache()
        else:
            logger.error(f"Error while processing video {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
        return None

def distribute_videos_to_gpus(video_ids, args):
    if not args.distributed:
        return video_ids
    
    videos_per_rank = len(video_ids) // args.world_size
    
    remainder = len(video_ids) % args.world_size
    
    start_idx = args.rank * videos_per_rank
    
    if args.rank < remainder:
        start_idx += args.rank
    else:
        start_idx += remainder
    
    end_idx = start_idx + videos_per_rank
    if args.rank < remainder:
        end_idx += 1
    
    assigned_videos = video_ids[start_idx:end_idx]
    
    logger.info(f"Process {args.rank}: processing {len(assigned_videos)} videos, indices from {start_idx} to {end_idx-1}")
    
    return assigned_videos

def save_checkpoint(generator, optimizer, args, total_processed, total_videos, loss=None):
    if args.distributed and args.rank != 0:
        return
        
    os.makedirs(args.save_path, exist_ok=True)
    
    filename = os.path.join(args.save_path, f'checkpoint_videos_{total_processed}.pth')
    
    state_dict = generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict()
    
    checkpoint = {
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'total_processed': total_processed,
        'total_videos': total_videos,
        'args': args
    }
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved to {filename} (Progress: {total_processed}/{total_videos})")
    
    latest_filename = os.path.join(args.save_path, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_filename)

def load_checkpoint(path, generator, optimizer, device):
    logger.info(f"Loading checkpoint from {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if hasattr(generator, 'module'):
        generator.module.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint['state_dict'])
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    total_processed = checkpoint.get('total_processed', 0)
    total_videos = checkpoint.get('total_videos', 0)
    loss = checkpoint.get('loss', None)
    
    return total_processed, total_videos, loss

def adjust_learning_rate(optimizer, total_processed, total_videos, args):
    if total_videos == 0:
        return args.initial_lr
    
    progress = min(total_processed / total_videos, 1.0)
    
    lr = args.initial_lr + progress * (args.final_lr - args.initial_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def batch_process_videos(generator, optimizer, args, tokenizer, model, feature_extractor, device, start_processed=0):
    all_qa_data = util.load_qa_data(args.qa_json)
    if not all_qa_data:
        logger.error("Error: JSON file is empty or invalid")
        return 0, 0
    
    video_ids = list(all_qa_data.keys())
    logger.info(f"Found {len(video_ids)} videos in JSON file")
    
    if args.max_videos > 0 and args.max_videos < len(video_ids):
        logger.info(f"Limiting processing to {args.max_videos} videos")
        video_ids = video_ids[:args.max_videos]
    
    video_ids.sort()
    total_videos = len(video_ids)
    
    total_loss = 0
    processed_count = 0
    skipped_count = 0
    total_processed = start_processed
    last_save_time = time.time()
    
    for i, video_id in enumerate(video_ids):
        current_lr = adjust_learning_rate(optimizer, total_processed, total_videos, args)
        
        if i % args.print_frequency == 0 or i == 0:
            logger.info(f"Current learning rate: {current_lr:.6f} (Progress: {total_processed}/{total_videos})")
        
        qa_pairs = all_qa_data[video_id]
        video_path = os.path.join(args.video_input_dir, f"{video_id}")
        
        try:
            logger.info(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
            
            logger.info(f"Loading frames for video {video_id}...")
            
            try:
                video_reader = VideoReader(video_path, num_threads=4)
                total_frames = len(video_reader)
                
                sample_rate = args.frame_sample_rate
                max_frames = 5000
                
                if sample_rate > 1:
                    frame_indices = np.arange(0, total_frames, sample_rate)
                else:
                    interval = max(1, total_frames // max_frames)
                    frame_indices = np.arange(0, total_frames, interval)
                
                if max_frames > 0 and len(frame_indices) > max_frames:
                    indices_subset = np.linspace(0, len(frame_indices)-1, max_frames, dtype=int)
                    frame_indices = frame_indices[indices_subset]
                
                frames = video_reader.get_batch(frame_indices).asnumpy()
                
                frames_tensor = torch.from_numpy(frames)
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)
                
                if frames_tensor.dtype == torch.uint8:
                    frames_tensor = frames_tensor.float() / 255.0
                

                image_size = 448
                size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
                frames_tensor = F.interpolate(
                        frames_tensor, 
                        size=size,
                        mode='bilinear', 
                        align_corners=False
                    )
                
                logger.info(f"Loaded {len(frames_tensor)} frames from video {video_id}")
                
                if len(frames_tensor) > 0:
                    current_video_loss = 0
                    for iter_idx in range(args.samples_per_gpu):
                        try:
                            if len(frames_tensor) <= args.keyframe_count:
                                key_frames = frames_tensor
                            else:
                                total_frames = len(frames_tensor)
                                base_stride = total_frames / args.keyframe_count
                                offset = (iter_idx * base_stride / args.samples_per_gpu) % 1.0
                                indices = [int(min(total_frames - 1, (i + offset) * base_stride)) for i in range(args.keyframe_count)]
                                indices = sorted(list(set(indices)))
                                
                                while len(indices) < args.keyframe_count:
                                    max_gap = 0
                                    insert_pos = 0
                                    for j in range(len(indices) - 1):
                                        gap = indices[j+1] - indices[j]
                                        if gap > max_gap:
                                            max_gap = gap
                                            insert_pos = j
                                    
                                    new_idx = indices[insert_pos] + max_gap // 2
                                    indices.insert(insert_pos + 1, new_idx)
                                
                                key_frames = torch.stack([frames_tensor[i] for i in indices])
                            
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                key_frames = key_frames.to(device).to(torch.bfloat16).contiguous()
                                
                                noise = generator(key_frames)
                                noise = torch.clamp(noise, -args.epsilon, args.epsilon)
                                
                                adv_frames = key_frames + noise
                                adv_frames = torch.clamp(adv_frames, 0, 1)
                                
                                orig_vlm_features = model.extract_feature(key_frames)
                                adv_vlm_features = model.extract_feature(adv_frames)
                                
                                loss = attack_utils.combined_adversarial_loss(
                                    orig_vlm_features, adv_vlm_features,
                                    key_frames, adv_frames,
                                    qa_pairs, tokenizer, model, feature_extractor, device, args
                                )
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            current_video_loss += loss.item()
                            
                            if iter_idx % 20 == 0 or iter_idx == args.samples_per_gpu - 1:
                                logger.info(f"Video {video_id} - Iteration {iter_idx+1}/{args.samples_per_gpu}, Loss: {loss.item():.4f}")
                            
                        except Exception as e:
                            if is_oom_error(e):
                                logger.warning(f"OOM error during iteration {iter_idx} for video {video_id}, skipping video")
                                log_oom_video(video_id, args)
                                torch.cuda.empty_cache()
                                raise
                            else:
                                logger.error(f"Error during iteration {iter_idx} for video {video_id}: {str(e)}")
                                logger.error(traceback.format_exc())
                                continue
                    
                    avg_video_loss = current_video_loss / args.samples_per_gpu
                    total_loss += avg_video_loss
                    processed_count += 1
                    total_processed += 1
                    
                    logger.info(f"Video {video_id} processing complete, average loss: {avg_video_loss:.4f}")
                
                else:
                    logger.error(f"Loaded 0 frames from video {video_id}, skipping")
                    skipped_count += 1
                
            except Exception as e:
                if is_oom_error(e):
                    logger.warning(f"OOM error while processing video {video_id}, skipping video")
                    log_oom_video(video_id, args)
                else:
                    logger.error(f"Error while processing video {video_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                skipped_count += 1
                continue
            
            del frames_tensor
            torch.cuda.empty_cache()
            
            if (processed_count % args.save_every_n_videos == 0 or 
                time.time() - last_save_time > 3600):
                
                current_avg_loss = total_loss / processed_count if processed_count > 0 else 0
                
                if args.distributed:
                    if args.rank == 0:
                        save_checkpoint(
                            generator, optimizer, args,
                            total_processed, total_videos,
                            current_avg_loss
                        )
                else:
                    save_checkpoint(
                        generator, optimizer, args,
                        total_processed, total_videos,
                        current_avg_loss
                    )
                
                last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Uncaught error while processing video {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            skipped_count += 1
            continue
    
    local_avg_loss = total_loss / processed_count if processed_count > 0 else 0
    
    logger.info(f"Processing complete, local average loss: {local_avg_loss:.4f}")
    logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Total: {total_processed}")
    
    if not args.distributed or args.rank == 0:
        save_checkpoint(generator, optimizer, args, total_processed, total_videos, local_avg_loss)
    
    return local_avg_loss, total_processed

def main():
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    
    init_distributed_mode(args)
    
    if args.distributed:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not args.distributed or args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)
    
    if not args.distributed or args.rank == 0:
        logger.info(f"Starting training, Distributed mode: {args.distributed}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Using device: {device}")
        logger.info(f"Loading model: {args.model_path}")
        logger.info(f"Iterations per video: {args.samples_per_gpu}")
        logger.info(f"Initial LR: {args.initial_lr}, Final LR: {args.final_lr}")
        logger.info(f"Saving checkpoint every {args.save_every_n_videos} videos")
        if args.use_feature_model:
            logger.info(f"Using combined loss (VLM features + Answer loss + ResNet features)")
        else:
            logger.info(f"Using VLM feature loss only")
            
    model, tokenizer = model_utils.load_model(args.model_path, device, args.token)
    for param in model.parameters():
        param.requires_grad = False
    
    feature_extractor = None
    if args.use_feature_model:
        if not args.distributed or args.rank == 0:
            logger.info(f"Loading feature extraction model: {'Pretrained ResNet50' if args.feature_model is None else args.feature_model}")
        
        model_c = vision_models.resnet50()
        
        if args.feature_model is not None:
            model_c.load_state_dict(torch.load(args.feature_model), strict=False)
        
        feature_extractor = torch.nn.Sequential(*list(model_c.children())[:-1])
        feature_extractor = feature_extractor.eval().to(device)
        
        for param in feature_extractor.parameters():
            param.requires_grad = False
    
    start_processed = 0
    
    if args.resume:
        generator = Generator().to(device).to(torch.bfloat16)
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            generator.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer = torch.optim.AdamW(generator.parameters(), lr=args.initial_lr)
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    else:
        generator = model_utils.Generator(
            factor=1.5,
            use_attention=True,
            use_checkpoint=False,
            base_channels=32
        ).to(device).to(torch.bfloat16)
        
        optimizer = torch.optim.AdamW(generator.parameters(), lr=args.initial_lr)

    if args.distributed:
        try:
            torch.distributed.barrier()
        except Exception as e:
            logger.warning(f"Initial barrier synchronization failed, but continuing: {str(e)}")
    
    if args.distributed:
        generator = DDP(generator, 
                        device_ids=[args.gpu], 
                        broadcast_buffers=False)
    
    try:
        avg_loss, total_processed = batch_process_videos(
            generator, optimizer, args, tokenizer, model, 
            feature_extractor, device, start_processed
        )
        
        if not args.distributed or args.rank == 0:
            final_model_path = os.path.join(args.save_path, 'generator_final.pth')
            torch.save({
                'state_dict': generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'total_processed': total_processed,
                'loss': avg_loss
            }, final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        
    except KeyboardInterrupt:
        if not args.distributed or args.rank == 0:
            logger.info("Training interrupted by user")
            interrupted_model_path = os.path.join(args.save_path, 'generator_interrupted.pth')
            torch.save({
                'state_dict': generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'total_processed': total_processed if 'total_processed' in locals() else start_processed,
                'interrupted': True
            }, interrupted_model_path)
            logger.info(f"Saved interrupted model to {interrupted_model_path}")
    except Exception as e:
        logger.error(f"Uncaught error during training: {str(e)}")
        logger.error(traceback.format_exc())
        if not args.distributed or args.rank == 0:
            error_model_path = os.path.join(args.save_path, 'generator_error.pth')
            try:
                torch.save({
                    'state_dict': generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict(),
                    'optimizer': optimizer.state_dict() if 'optimizer' in locals() else None,
                    'total_processed': total_processed if 'total_processed' in locals() else start_processed,
                    'error': str(e)
                }, error_model_path)
                logger.info(f"Saved error state model to {error_model_path}")
            except:
                logger.error("Failed to save error state model")

if __name__ == "__main__":
    main()