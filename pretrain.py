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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import traceback
import logging
import cv2
from model_utils import Generator
import glob
import webdataset as wds

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
import logging
logging.getLogger().setLevel(logging.WARNING)

import model_utils
import attack_utils
import config
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_attack")

def parse_args():
    parser = argparse.ArgumentParser(description='image attack with WebDataset support')
    parser.add_argument('--model_path', type=str, default='.cache/models--OpenGVLab--InternVL2_5-1B/snapshots/ec696efff1e392b75c46ef1710017ea6244423d1', help='model path')
    parser.add_argument('--output_dir', type=str, default=config.DEFAULT_OUTPUT_DIR, help='output directory')
    parser.add_argument('--epsilon', type=float, default=config.DEFAULT_EPSILON, help='maximum perturbation size (L∞ norm)')
    parser.add_argument('--print_frequency', type=int, default=10, help='frequency for printing progress')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token for model download')
    parser.add_argument('--image_input_dir', type=str, default='/shared/hf-hub/datasets--laion-400m/laion400m-data', help='WebDataset tar files directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='skip already processed images')
    parser.add_argument('--save_path', type=str, default='./weights/laion/', help='generator weights save path')
    parser.add_argument('--save_freq', type=int, default=1, help='frequency for saving weights (every n epochs)')

    parser.add_argument('--semantic_loss', type=float, default=0.1)
    parser.add_argument('--feature_loss', type=float, default=20.0)
    parser.add_argument('--smooth_loss', type=float, default=10.0)

    parser.add_argument('--use_feature_model', action='store_true', default=config.USE_FEATURE_MODEL, help='use additional feature model')
    parser.add_argument('--feature_model', type=str, default=config.DEFAULT_FEATURE_MODEL, help='feature model weights path (default: pretrained ResNet50)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1)

    parser.add_argument('--use_data_augmentation', action='store_true', default=True, help='use data augmentation')
    parser.add_argument('--image_size', type=int, default=448, help='processed image size')

    parser.add_argument('--dist-url', '--dist_url', default='env://', type=str, help='URL for distributed training')
    parser.add_argument('--dist-backend', '--dist_backend', default='nccl', type=str, help='backend for distributed training')
    parser.add_argument('--world-size', '--world_size', default=-1, type=int, help='number of processes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='global rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU ID to use')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1, help='node-local process rank')
    parser.add_argument('--multiprocessing-distributed', '--multiprocessing_distributed', action='store_true',
                        help='use multi-process distributed training, suitable for multi-node training')
    
    parser.add_argument('--fixed_question', type=str, default="Describe what you see in this image.", 
                       help='fixed question for all image descriptions')
    parser.add_argument('--webdataset_shuffle_size', type=int, default=5000, help='WebDataset random buffer size')

    parser.add_argument('--resume', type=str, help='checkpoint path to resume training')

    
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
        rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def get_webdataset_loader(args, is_distributed=False):
    if args.use_data_augmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
    
    tar_files = glob.glob(os.path.join(args.image_input_dir, "*.tar"))
    
    if not tar_files:
        logger.error(f"No tar files found in {args.image_input_dir}")
        return None, None
    
    logger.info(f"Found {len(tar_files)} tar files in {args.image_input_dir}")
    
    dataset = (
        wds.WebDataset(tar_files, resampled=True, shardshuffle=True, nodesplitter=wds.shardlists.split_by_node)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map(lambda x: (transform(x[0]), x[1]))
        .batched(args.batch_size)
    )
    
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, None

def calculate_adversarial_deltas(images, qa_pairs, tokenizer, model, feature_extractor, device, args, generator, optimizer):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        images = images.to(device).to(torch.bfloat16).contiguous()
        noise = generator(images)
        noise = torch.clamp(noise, -args.epsilon, args.epsilon)
        adv_images = images + noise
        adv_images = torch.clamp(adv_images, 0, 1)
        
        orig_vlm_features = model.extract_feature(images)
        adv_vlm_features = model.extract_feature(adv_images)
        loss = attack_utils.combined_adversarial_loss(
                orig_vlm_features, adv_vlm_features,
                images, adv_images,
                qa_pairs, tokenizer, model, feature_extractor, device, args
            )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def run_adversarial_attack(images, image_ids, qa_pairs, tokenizer, model, feature_extractor, device, args, generator, optimizer):
    loss = calculate_adversarial_deltas(
        images, qa_pairs, tokenizer, model, feature_extractor, device, args, generator, optimizer
    )
    
    return loss

def batch_process_webdataset(epoch, generator, optimizer, args, tokenizer, model, feature_extractor, device):
    data_loader, _ = get_webdataset_loader(args, args.distributed)
    
    if data_loader is None:
        logger.error("Error: Failed to create WebDataset loader")
        return 0
    
    total_loss = 0
    processed_batches = 0
    
    for i, (images, texts) in enumerate(data_loader):
        images = images.to(device)
        
        batch_qa_pairs = []
        for text in texts:
            batch_qa_pairs.append({
                'question': args.fixed_question,
                'answer': text.strip()
            })
        
        image_ids = [f"laion_{epoch}_{i}_{j}" for j in range(len(images))]
        
        loss = run_adversarial_attack(
            images, image_ids, batch_qa_pairs, tokenizer, model,
            feature_extractor, device, args, generator, optimizer
        )
        

        total_loss += loss
        processed_batches += 1
        
        if i % args.print_frequency == 0:
            logger.info(f"Epoch: {epoch}, Rank: {args.rank if args.distributed else 0}, "
                        f"Batch: {i}, Loss: {loss:.4f}")
        if i % 1000 == 0:
            save_generator(generator, epoch, args, loss)
                    
    
    local_avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    
    if args.distributed:
        world_size = args.world_size
        all_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
        local_loss_tensor = torch.tensor([local_avg_loss]).to(device)
        
        dist.all_gather(all_losses, local_loss_tensor)
        
        global_avg_loss = sum([loss.item() for loss in all_losses]) / world_size
    else:
        global_avg_loss = local_avg_loss
    
    if not args.distributed or args.rank == 0:
        logger.info(f"Epoch {epoch} completed, global average loss: {global_avg_loss:.4f}")
        logger.info(f"Successfully processed batches: {processed_batches}")
    
    return global_avg_loss

def save_generator(generator, epoch, args, loss=None, is_best=False):
    if args.distributed and args.rank != 0:
        return
        
    checkpoint = {
        'epoch': epoch,
        'state_dict': generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict(),
        'args': args,
    }
    if loss is not None:
        checkpoint['loss'] = loss
        
    filename = os.path.join(args.save_path, f'generator_epoch_{epoch}.pth')
    torch.save(checkpoint, filename)
    logger.info(f"Saved model weights to {filename}")
    
    if is_best:
        best_filename = os.path.join(args.save_path, 'generator_best.pth')
        torch.save(checkpoint, best_filename)
        logger.info(f"Saved best model weights to {best_filename}")

def main():
    args = parse_args()
    config.ANSWER_LOSS_WEIGHT = args.semantic_loss
    config.VLM_FEATURE_WEIGHT = args.feature_loss
    config.RESNET_FEATURE_WEIGHT = args.smooth_loss
    
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
        logger.info(f"Starting training, distributed mode: {args.distributed}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Using device: {device}")
        logger.info(f"Loading model: {args.model_path}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Image size: {args.image_size}x{args.image_size}")
        logger.info(f"Using WebDataset to load LAION-400M dataset")
        logger.info(f"Fixed question: '{args.fixed_question}'")
            
        if args.use_feature_model:
            logger.info(f"Using combined loss (VLM Feat + Answer Loss + ResNet Feat)")
        else:
            logger.info(f"Using only VLM feature loss")
            
    model, tokenizer = model_utils.load_model(args.model_path, device, args.token)
    for param in model.parameters():
        param.requires_grad = False

    feature_extractor = None
    if args.use_feature_model:
        if not args.distributed or args.rank == 0:
            logger.info(f"Loading feature extraction model: {'pretrained ResNet50' if args.feature_model is None else args.feature_model}")
        
        model_c = vision_models.resnet50()
        
        if args.feature_model is not None:
            model_c.load_state_dict(torch.load(args.feature_model), strict=False)
        
        feature_extractor = torch.nn.Sequential(*list(model_c.children())[:-1])
        feature_extractor = feature_extractor.eval().to(device)
        
        for param in feature_extractor.parameters():
            param.requires_grad = False
    
    generator = Generator().to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr*0.01)

    if os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if args.distributed:
        generator = DDP(generator, 
                       device_ids=[args.gpu], 
                       broadcast_buffers=False)
    
    
    if args.distributed:
        torch.distributed.barrier()
    
    start_epoch = 0
    best_loss = float('inf')
    

    for epoch in range(start_epoch, args.epoch):
        if not args.distributed or args.rank == 0:
            logger.info(f"Starting Epoch {epoch}/{args.epoch-1}")
        
        if args.distributed:
            torch.distributed.barrier()
        
        avg_loss = batch_process_webdataset(
            epoch, generator, optimizer, args, tokenizer, model, feature_extractor, device
        )
        
        scheduler.step()
        if (epoch + 1) % args.save_freq == 0:
            save_generator(generator, epoch, args, avg_loss)
    
    if not args.distributed or args.rank == 0:
        final_model_path = os.path.join(args.save_path, 'generator_final.pth')
        torch.save({
            'epoch': args.epoch - 1,
            'state_dict': generator.module.state_dict() if hasattr(generator, 'module') else generator.state_dict(),
            'args': args,
            'loss': avg_loss
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    main()