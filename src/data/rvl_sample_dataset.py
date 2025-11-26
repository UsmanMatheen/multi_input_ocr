import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

class RVLSampleDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_file"]
        
        img = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            img = self.transform(img)

        return img, label
