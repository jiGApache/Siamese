import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from Models.SiameseModel import Siamese
from Datasets.PairDataset import NormVSAll as PairDataset


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)


def contrastive_loss(emb_1, emb_2, y):

    distances = torch.zeros(len(y), dtype=torch.float32)
    for i in range(len(y)):
        if y[i] == 1:
            distances[i] = torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2))
        else:
            distances[i] = torch.maximum(torch.as_tensor(
                0.), 1.0 - torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2)))

    loss = torch.mean(distances)

    return loss


def train_epoch(
        epoch_number: int,
        model: Siamese,
        dataloader: DataLoader,
        optimizer: torch.optim.Adam
):

    steps_in_epoch = 0
    epoch_loss = 0.0

    with tqdm(range(int(len(dataloader.dataset) / hyper_params['train_params']['batch_size'])), desc=f'Epoch {epoch_number}', ncols=100) as pbar:
        for pair_ecg, label in dataloader:
            steps_in_epoch += 1

            ecgs1, ecgs2, label = pair_ecg[0].to(
                hyper_params['train_params']['device'], non_blocking=True), pair_ecg[1].to(
                hyper_params['train_params']['device'], non_blocking=True), label.to(
                hyper_params['train_params']['device'], non_blocking=True)
            out_emb_1, out_emb_2 = model(ecgs1, ecgs2)

            loss = hyper_params['train_params']['loss_function'](
                out_emb_1, out_emb_2, label)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del out_emb_1, out_emb_2, loss

            pbar.update(1)

    return epoch_loss / steps_in_epoch


def val_epoch(
        epoch_number: int,
        model: Siamese,
        dataloader: DataLoader
):

    steps_in_epoch = 0
    epoch_loss = 0.0

    with tqdm(range(int(len(dataloader.dataset) / hyper_params['train_params']['batch_size'])), desc=f'Epoch {epoch_number}', ncols=100) as pbar:
        for pair_ecg, label in dataloader:
            steps_in_epoch += 1

            ecgs1, ecgs2, label = pair_ecg[0].to(
                hyper_params['train_params']['device'], non_blocking=True), pair_ecg[1].to(
                hyper_params['train_params']['device'], non_blocking=True), label.to(
                hyper_params['train_params']['device'], non_blocking=True)
            out_emb_1, out_emb_2 = model(ecgs1, ecgs2)

            epoch_loss += hyper_params['train_params']['loss_function'](
                out_emb_1, out_emb_2, label).item()

            del out_emb_1, out_emb_2

            pbar.update(1)

    return epoch_loss / steps_in_epoch


def train_model() -> float:

    train_ds = PairDataset(
        abnorm_labels=hyper_params['train_params']['abnorm_labels'],
        folder='Train',
        samples_per_element=2)
    val_ds = PairDataset(
        abnorm_labels=hyper_params['train_params']['abnorm_labels'],
        folder='Val',
        samples_per_element=10)
    train_dl = DataLoader(
        train_ds,
        batch_size=hyper_params['train_params']['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True)
    val_dl = DataLoader(
        val_ds,
        batch_size=hyper_params['train_params']['batch_size'],
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        drop_last=True)

    if not os.path.exists('./ECG_embedding_classification/Nets'):
        os.mkdir('./ECG_embedding_classification/Nets')

    b_loss = float('inf')

    model = Siamese(
        kernel_size=hyper_params['model_params']['kernel_size'],
        num_features=hyper_params['model_params']['num_features'],
        like_LU_func=hyper_params['model_params']['activation_function'],
        norm1d=hyper_params['model_params']['normalization'],
        dropout_rate=hyper_params['model_params']['dropout_rate']
    ).to(hyper_params['train_params']['device'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper_params['train_params']['initial_rate'],
        weight_decay=hyper_params['train_params']['weight_decay'])

    for epoch in range(hyper_params['train_params']['epochs']):

        model.train(True)
        train_loss = train_epoch(epoch + 1, model, train_dl, optimizer,)
        torch.cuda.empty_cache()

        model.train(False)
        with torch.no_grad():
            test_loss = val_epoch(epoch + 1, model, val_dl)
        torch.cuda.empty_cache()

        print(
            f'Epoch: {epoch+1} Train loss: {train_loss:.5f} Validation loss:  {test_loss:.5f}\n\n')

        if b_loss > test_loss:
            torch.save(model.state_dict(),
                       f'./ECG_embedding_classification/Nets/embedding_extractor.pth')
            b_loss = test_loss
        else:
            break

        for g in optimizer.param_groups:
            g['lr'] /= hyper_params['train_params']['learning_step']

    return b_loss


hyper_params = {
    'train_params': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'initial_rate': 0.00001,
        'learning_step': 1.0,
        'abnorm_labels': ['STTC', 'MI', 'HYP', 'CD'],
        'loss_function': contrastive_loss,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'epochs': 1
    },
    'model_params': {
        'kernel_size': 32,
        'num_features': 92,
        'activation_function': torch.nn.GELU,
        'normalization': torch.nn.BatchNorm1d,
        'dropout_rate': 0.2
    }
}

if __name__ == '__main__':

    set_global_seed(42)
    train_model()
