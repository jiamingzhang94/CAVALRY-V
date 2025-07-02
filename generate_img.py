import os
import argparse
import io
import base64

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_utils import Generator

class ImageFolder(Dataset):
    def __init__(self, folder_path, resize=224):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ])
        self.base_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        path = os.path.join(self.folder_path, img_name)
        with Image.open(path) as img:
            original_size = img.size
            if img.mode == 'L':
                img = img.convert('RGB')
            resized = self.transform(img)
            original = self.base_transform(img)
        return resized, original, img_name, original_size


def save_image(tensor, path):
    tensor = tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    if tensor.ndim == 2 or tensor.shape[2] == 1:
        tensor = tensor.squeeze(-1)
    img = Image.fromarray(np.clip(tensor * 255, 0, 255).astype(np.uint8))
    img.save(path)


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def mme_to_tsv(raw_tsv, image_root, output_tsv):
    df = pd.read_csv(raw_tsv, sep='\t').drop(columns=['image'])
    imgs = []
    for idx in tqdm(df['index'], desc='Encoding images'):
        path = os.path.join(image_root, f'{idx}.jpg')
        imgs.append(encode_image_to_base64(Image.open(path)))
    df['image'] = imgs
    df.to_csv(output_tsv, sep='\t', index=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    gen = Generator().to(device).to(torch.float32)
    ckpt = torch.load(args.generator_path, map_location=device)
    gen.load_state_dict(ckpt['state_dict'])
    gen.eval()

    dataset = ImageFolder(args.input_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for resized, original, names, sizes in tqdm(loader, desc='Processing'):
        resized, original = resized.to(device), original.to(device)
        h, w = original.shape[2:]
        noise = torch.clamp(gen(resized), -args.epsilon, args.epsilon)
        noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)
        perturbed = torch.clamp(original + noise, 0.0, 1.0)
        save_image(perturbed, os.path.join(args.output_dir, names[0]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--raw_tsv', required=True)
    p.add_argument('--output_tsv', required=True)
    p.add_argument('--generator_path', required=True)
    p.add_argument('--epsilon', type=float, default=0.0627)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    mme_to_tsv(args.raw_tsv, args.output_dir, args.output_tsv)
