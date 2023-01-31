import os
import collections

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as alb

from src.utils import get_config

global_config = get_config()


class ImgTextDataset(Dataset):
    def __init__(self, image_dir, image_filenames, captions, tokenizer):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.captions = captions
        self.encoded_captions = tokenizer(
            captions.tolist(),
            padding=True, truncation=True,
            return_tensors='pt',
            max_length=global_config['model_config']['text_model']['max_length'])
        self.tokenizer = tokenizer
        self.transforms = self._data_transforms()

    def __getitem__(self, index):
        #return_item = collections.namedtuple("item", ["image", "caption", "input_ids", "attention_masks"])
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        # using cv2 for image reading which is claimed to be faster than PIL based pytorch one
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item = dict({
            "image": torch.tensor(image).permute(2, 0, 1).float(),
            "caption": self.captions[index],
            "input_ids": self.encoded_captions['input_ids'][index],
            "attention_masks": self.encoded_captions['attention_mask'][index]
        })
        return item

    def __len__(self):
        #assert len(self.captions) == len(self.image_filenames)
        return len(self.image_filenames)

    @staticmethod
    def _data_transforms():
        """albumentations performs better than torch transforms"""
        transformations = alb.Compose(
            [
                alb.Resize(global_config['data_config']['image_size'],
                           global_config['data_config']['image_size'],
                           always_apply=True),
                alb.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
        return transformations
