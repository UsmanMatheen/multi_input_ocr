import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd


class RVLMultiInputDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_transform, tokenizer, max_length=256):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(self.csv_path)

        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = self.img_dir / row["image_file"]
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # Text (OCR)
        text = str(row["ocr_text"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
