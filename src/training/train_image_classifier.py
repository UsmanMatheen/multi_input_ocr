import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from src.data.rvl_sample_dataset import RVLSampleDataset
from src.models.image_classifier import ImageClassifier

from pathlib import Path


def train_model(sample_dir, epochs=3, batch_size=8, lr=1e-4, device="cpu"):
    sample_dir = Path(sample_dir)
    csv_path = sample_dir / "labels.csv"
    img_dir = sample_dir

    # Transformations (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = RVLSampleDataset(csv_path, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImageClassifier(num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = running_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    # save checkpoint
    ckpt_path = sample_dir / "resnet18_sample.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to: {ckpt_path}")
