import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class TextClassifier(nn.Module):
    def __init__(self, num_labels=16, model_name="distilbert-base-uncased"):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # outputs has: loss (if labels), logits, hidden_states (optional)
        return outputs
