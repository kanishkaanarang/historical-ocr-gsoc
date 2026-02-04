# dataset/ocr_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from utils.charset import char_to_idx

class OCRDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(labels_file, "r") as f:
            self.samples = [line.strip().split(",") for line in f]

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return torch.tensor([char_to_idx[c] for c in text])

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label_encoded = self.encode_text(label)

        return image, label_encoded
