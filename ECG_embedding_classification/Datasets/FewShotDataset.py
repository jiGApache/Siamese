import os
import torch
import scipy.io
import numpy as np
import pandas as pd
from Datasets.DatasetPrepocessing import prepare_data

class FewShotDataset:

    def __init__(
        self,
        abnormal_labels=['STTC', 'MI', 'HYP', 'CD'],
        shot=3
    ):

        np.random.seed(42)

        if not self.is_data_ready(
                path='./ECG_embedding_classification/Data/',
                folder='Test'):
            prepare_data(path='./ECG_embedding_classification/Data/')

        self.FRAGMENT_SIZE = 4000
        df = pd.read_csv('./ECG_embedding_classification/Data/Test/LOCAL_REFERENCE.csv')
        normal_col = 'NORM'

        normal_df = df.loc[
            (df[normal_col] == 1) &
            (df['STTC'] == 0) &
            (df['MI'] == 0) &
            (df['HYP'] == 0) &
            (df['CD'] == 0)
        ].reset_index(drop=True)

        abnormal_dfs = []
        for col in abnormal_labels:
            abnormal_dfs.append(
                df.loc[
                    (df[normal_col] == 0) &
                    (df[col] == 1)
                ].reset_index(drop=True)
            )

        self.data = []

        for i in range(len(normal_df)):
            self.data.append((
                f'NormFiltered/{str(normal_df["ecg_id"][i]).zfill(5)}.mat',
                1
            ))
        for df in abnormal_dfs:
            for i in range(len(df)):
                self.data.append((
                    f'NormFiltered/{str(df["ecg_id"][i]).zfill(5)}.mat',
                    0
                ))

        self.norm_indices, self.abnorm_indices = list(
            range(0, len(normal_df))), list(range(len(normal_df), len(self.data)))
        self.norm_indices_shots = []
        self.abnorm_indices_shots = []

        for _ in range(shot):
            self.norm_indices_shots.append(self.norm_indices.pop(
                np.random.randint(low=0, high=len(self.norm_indices))))
            self.abnorm_indices_shots.append(self.abnorm_indices.pop(
                np.random.randint(low=0, high=len(self.abnorm_indices))))

    def get_train_item(self, index):

        if index < len(self.norm_indices_shots):
            file_name, label = self.data[self.norm_indices_shots[index]]
            return (torch.as_tensor(scipy.io.loadmat(f'./ECG_embedding_classification/Data/Test/{file_name}')[
                'ECG'][:, :self.FRAGMENT_SIZE], dtype=torch.float32)[None, :, :], label)
        else:
            index -= len(self.norm_indices_shots)
            file_name, label = self.data[self.abnorm_indices_shots[index]]
            return (torch.as_tensor(scipy.io.loadmat(f'./ECG_embedding_classification/Data/Test/{file_name}')[
                'ECG'][:, :self.FRAGMENT_SIZE], dtype=torch.float32)[None, :, :], label)

    def get_test_item(self, index):

        if index < len(self.norm_indices):
            file_name, label = self.data[self.norm_indices[index]]
            return (
                torch.as_tensor(scipy.io.loadmat(
                    f'./ECG_embedding_classification/Data/Test/{file_name}')['ECG'][:, :4000], dtype=torch.float32)[None, :, :],
                label
            )
        else:
            index -= len(self.norm_indices)
            file_name, label = self.data[self.abnorm_indices[index]]
            return (
                torch.as_tensor(scipy.io.loadmat(
                    f'./ECG_embedding_classification/Data/Test/{file_name}')['ECG'][:, :4000], dtype=torch.float32)[None, :, :],
                label
            )

    def get_train_size(self):
        return len(self.norm_indices_shots) + len(self.abnorm_indices_shots)

    def get_test_size(self):
        return len(self.norm_indices) + len(self.abnorm_indices)
    
    def is_data_ready(self, path, folder):
        return os.path.exists(f'{path}/{folder}/NormFiltered') \
            and len(os.listdir(f'{path}/{folder}/NormFiltered')) != 0
