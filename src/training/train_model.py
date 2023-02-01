import numpy as np
import pandas as pd
import tqdm
from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from src.data_utils.dataset import ImgTextDataset
from src.utils import get_config
from src.utils import AverageMeter
from src.models.clip_model import CLIPModel

global_config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_train_validation_data():
    captions_file = global_config['data_config']['captions_file']
    captions_df = pd.read_csv(captions_file, sep=",")

    mask = np.random.rand(len(captions_df)) < 0.8
    train_data = captions_df[mask]
    validation_data = captions_df[~mask]
    return train_data, validation_data


def build_dataloaders(data: pd.DataFrame, tokenizer, batch_size, shuffle=True):
    images_path = global_config['data_config']['images_dir']
    image_files = data['image'].values
    captions = data['caption'].values

    dataset = ImgTextDataset(images_path, image_files, captions, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def run_single_epoch(model, loader, optimizer, lr_scheduler=None, mode_validation=False, step_epoch=True):
    loss_meter = AverageMeter("loss_meter")
    for _, batch in enumerate(pbar := tqdm.tqdm(loader, total=len(loader))):
        # push data to device to avoid mismatch
        batch = {k: v.to(device) for k,v in batch.items() if k!='caption'}
        loss = model(batch)
        if not mode_validation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not step_epoch:
                lr_scheduler.step()
        # batch size
        count = batch['image'].size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(train_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, validation_df = generate_train_validation_data()
    tokenizer = DistilBertTokenizer.from_pretrained(
        global_config['model_config']['text_model']['text_tokenizer'])
    train_loader = build_dataloaders(train_df, tokenizer, batch_size=global_config['training_config']['batch_size'])
    validation_loader = build_dataloaders(validation_df, tokenizer, batch_size=global_config['training_config']['batch_size'])

    model = CLIPModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=global_config['training_config']['lr'],
        weight_decay=global_config['training_config']['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=global_config['training_config']['patience'],
        factor=global_config['training_config']['factor']
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(global_config['training_config']['epochs']):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = run_single_epoch(model, train_loader, optimizer, lr_scheduler)
        model.eval()
        with torch.no_grad():
            valid_loss = run_single_epoch(model, validation_loader, optimizer, None, mode_validation=True)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
