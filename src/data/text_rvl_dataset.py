import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class TextRVLDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
