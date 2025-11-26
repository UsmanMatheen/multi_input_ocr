import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizerFast

from pathlib import Path

from src.data.rvl_multi_input_dataset import RVLMultiInputDataset
from src.models.multi_input_classifier import MultiInputClassifier


def train_multi_input_model(
    sample_dir,
    epochs=3,
    batch_size=8,
    lr=1e-4,
    max_length=256,
    device="cpu",
):
    sample_dir = Path(sample_dir)
    csv_path = sample_dir / "labels_with_text.csv"
    img_dir = sample_dir

    # Image transforms (same as before)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    dataset = RVLMultiInputDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiInputClassifier(num_labels=16, freeze_backbones=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(
                image=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataloader)
        acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    ckpt_path = sample_dir / "multi_input_sample.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved multi-input model checkpoint to: {ckpt_path}")
