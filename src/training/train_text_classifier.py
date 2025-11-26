import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
import torch.optim as optim

from pathlib import Path

from src.data.text_rvl_dataset import TextRVLDataset
from src.models.text_classifier import TextClassifier


def train_text_model(
    sample_dir,
    epochs=3,
    batch_size=8,
    lr=2e-5,
    max_length=256,
    device="cpu",
):
    sample_dir = Path(sample_dir)
    csv_path = sample_dir / "labels_with_text.csv"

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    dataset = TextRVLDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextClassifier(num_labels=16).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataloader)
        acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    ckpt_path = sample_dir / "distilbert_text_sample.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved text model checkpoint to: {ckpt_path}")
