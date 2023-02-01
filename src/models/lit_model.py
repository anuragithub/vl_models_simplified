from torch import nn
import torch
from src.utils import get_config
from src.models.modules import TextTower, ImageTower, ProjectionHead

model_config = get_config()['model_config']


class LiTModel(nn.Module):
    def __init__(self,
                 image_embedding_size=model_config['image_model']['image_embedding'],
                 text_embedding_size=model_config['text_model']['text_embedding'],
                 temp_coefficient=model_config['temperature']):
        super().__init__()
        self.image_encoder = ImageTower()
        self.text_encoder = TextTower()
        self.image_projection_head = ProjectionHead(embedding_dim=image_embedding_size)
        self.text_projection_head = ProjectionHead(embedding_dim=text_embedding_size)
        self.image_embedding_size = image_embedding_size
        self.text_embedding_size = text_embedding_size
        self.temp_coefficient = temp_coefficient

    def forward(self, batch):
        # extract image, text features
        assert(batch.image.shape[0] == batch.captions.shape[0])
        image_features = self.image_encoder(batch.image)
        text_features = self.text_encoder(batch.captions)

        # image and text embeddings of common shape for logits
        image_representations = self.image_projection_head(image_features)
        text_representations = self.text_projection_head(text_features)

        logits = (image_representations @ text_representations.T)*torch.exp(self.temp_coefficient)

        labels = torch.arange(batch.image.shape[0])
        texts_loss = nn.CrossEntropyLoss(logits, labels, reduction='none')
        images_loss = nn.CrossEntropyLoss(logits.T, labels.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()


