import os
import torch
import scipy
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset
from Datasets.DatasetPrepocessing import prepare_data


class NormVSAll(Dataset):

    def __init__(self,
                 abnorm_labels=['STTC', 'MI', 'HYP', 'CD'],
                 samples_per_element=4,
                 folder='Train') -> None:
        super().__init__()

        if len(abnorm_labels) < 1:
            raise ValueError('The number of abnormal labels should be at least 1')

        self.folder = folder
        if not self.is_data_ready(
                path='./ECG_embedding_classification/Data/',
                folder=folder):
            prepare_data(path='./ECG_embedding_classification/Data/')

        df = pd.read_csv(
            f'./ECG_embedding_classification/Data/{folder}/LOCAL_REFERENCE.csv',
            index_col=None)

        self.FRAGMENT_SIZE = 4000
        self.normal_col = 'NORM'
        self.abnormal_col = abnorm_labels

        self.normal_data = []
        normal_df = df.loc[
            (df[self.normal_col] == 1) &
            (df['STTC'] == 0) &
            (df['MI'] == 0) &
            (df['HYP'] == 0) &
            (df['CD'] == 0)
        ].reset_index(drop=True)
        for i in range(len(normal_df)):
            self.normal_data.append(
                f'NormFiltered/{str(normal_df["ecg_id"][i]).zfill(5)}.mat')

        self.abnormal_data = []
        for col in self.abnormal_col:
            abnorm_d = []
            abnormal_df = df.loc[
                (df[self.normal_col] == 0) &
                (df[col] == 1)
            ].reset_index(drop=True)
            for i in range(len(abnormal_df)):
                abnorm_d.append(
                    f'NormFiltered/{str(abnormal_df["ecg_id"][i]).zfill(5)}.mat')
            self.abnormal_data.append(abnorm_d)

        self.samples_per_normal = samples_per_element * len(self.abnormal_data)
        self.samples_per_abnormal = samples_per_element

        self.normal_indices = np.random.choice(
            len(self.normal_data), len(self.normal_data), replace=False)
        self.normal_indices = np.tile(self.normal_indices, self.samples_per_normal)
        self.abnormal_indices = [np.random.choice(len(abn_d), len(
            abn_d), replace=False) for abn_d in self.abnormal_data]

        self.ds_len = int(len(self.normal_data) *
                          self.samples_per_normal +
                          len(self.normal_data) *
                          len(self.abnormal_data) *
                          self.samples_per_abnormal)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        # Pairs with equal normal labels
        if index < len(self.normal_data) * self.samples_per_normal:

            f_index = index // self.samples_per_normal
            s_index = self.normal_indices[index]

            ecg1 = scipy.io.loadmat(
                f'./ECG_embedding_classification/Data/{self.folder}/{self.normal_data[f_index]}')['ECG'][:, :self.FRAGMENT_SIZE]
            ecg2 = scipy.io.loadmat(
                f'./ECG_embedding_classification/Data/{self.folder}/{self.normal_data[s_index]}')['ECG'][:, :self.FRAGMENT_SIZE]

            return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32)
            ), torch.as_tensor((1.), dtype=torch.float32)

        else:
            index -= len(self.normal_data) * self.samples_per_normal

        # Pairs with different labels (norm & abnorm)
        f_index = index // self.samples_per_normal
        abnormal_d = self.abnormal_data[index % len(self.abnormal_data)]
        indices = self.abnormal_indices[index % len(self.abnormal_data)]
        s_index = indices[index % len(indices)]

        ecg1 = scipy.io.loadmat(
            f'./ECG_embedding_classification/Data/{self.folder}/{self.normal_data[f_index]}')['ECG'][:, :self.FRAGMENT_SIZE]
        ecg2 = scipy.io.loadmat(
            f'./ECG_embedding_classification/Data/{self.folder}/{abnormal_d[s_index]}')['ECG'][:, :self.FRAGMENT_SIZE]

        return (
            torch.as_tensor(ecg1, dtype=torch.float32),
            torch.as_tensor(ecg2, dtype=torch.float32)
        ), torch.as_tensor((0.), dtype=torch.float32)

    def is_data_ready(self, path, folder):
        return os.path.exists(f'{path}/{folder}/NormFiltered') \
            and len(os.listdir(f'{path}/{folder}/NormFiltered')) != 0

    def __len__(self):
        return self.ds_len
