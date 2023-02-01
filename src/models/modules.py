import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
from src.utils import get_config

global_config = get_config()
model_config = global_config['model_config']
image_tower_config = global_config['model_config']['image_model']
text_tower_config = global_config['model_config']['text_model']


class ImageTower(nn.Module):
    """
    Image tower for CLIP
    """

    def __init__(
        self, model_name=image_tower_config['model_name'],
            pretrained=image_tower_config['pretrained'],
            trainable=image_tower_config['trainable']
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextTower(nn.Module):
    def __init__(
            self, model_name=text_tower_config['model_name'],
            pretrained=text_tower_config['pretrained'],
            trainable=text_tower_config['trainable']):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # strategy employed in BERT that uses first token as sequence embedding
        # for more info refer: http://mccormickml.com/assets/BERT/CLS_token_500x606.png
        # https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=model_config['projection_dim'],
            dropout=model_config['dropout']
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x