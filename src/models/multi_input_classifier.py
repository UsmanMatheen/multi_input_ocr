import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel


class MultiInputClassifier(nn.Module):
    def __init__(
        self,
        num_labels=16,
        image_model_weights=models.ResNet18_Weights.IMAGENET1K_V1,
        text_model_name="distilbert-base-uncased",
        freeze_backbones=True,
    ):
        super().__init__()

        # Image backbone (ResNet18 without final fc)
        self.image_backbone = models.resnet18(weights=image_model_weights)
        img_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()  # output: [B, img_feat_dim]

        # Text backbone (DistilBERT without classifier head)
        self.text_backbone = DistilBertModel.from_pretrained(text_model_name)
        text_feat_dim = self.text_backbone.config.hidden_size  # usually 768

        # Optional: freeze backbones for small dataset
        if freeze_backbones:
            for p in self.image_backbone.parameters():
                p.requires_grad = False
            for p in self.text_backbone.parameters():
                p.requires_grad = False

        fusion_dim = img_feat_dim + text_feat_dim  # e.g., 512 + 768 = 1280

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )

    def forward(self, image, input_ids, attention_mask):
        # Image branch
        img_feat = self.image_backbone(image)  # [B, img_feat_dim]

        # Text branch - use first token embedding as sentence representation
        text_outputs = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [B, text_feat_dim]

        # Concatenate features
        fused = torch.cat([img_feat, text_feat], dim=1)

        logits = self.classifier(fused)
        return logits
