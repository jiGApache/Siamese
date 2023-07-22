import sys
import torch
import getopt
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from Models.EmbeddingModel import EmbeddingModule


def norm_vs_all(
        dim_reduction_method: str,
        data_type=str
):

    embedding_model = EmbeddingModule(
        kernel_size=hyper_params['model_params']['kernel_size'],
        num_features=hyper_params['model_params']['num_features'],
        like_LU_func=hyper_params['model_params']['activation_function'],
        norm1d=hyper_params['model_params']['normalization'],
        dropout_rate=hyper_params['model_params']['dropout_rate']
    )
    embedding_model.load_state_dict(torch.load(
        f'./ECG_embedding_classification/Nets/embedding_extractor.pth'))
    embedding_model.train(False)

    normal_ECGs = []
    abnormal_ECGs = []

    df = pd.read_csv(
        f'./ECG_embedding_classification/Data/{data_type}/LOCAL_REFERENCE.csv',
        delimiter=',',
        index_col=None)

    normal_df = df.loc[
        (df['NORM'] == 1) &
        (df['STTC'] == 0) &
        (df['MI'] == 0) &
        (df['HYP'] == 0) &
        (df['CD'] == 0)
    ].reset_index(drop=True)

    for i in range(len(normal_df)):
        normal_ECGs.append(
            torch.as_tensor(
                scipy.io.loadmat(
                    f'ECG_embedding_classification/Data/{data_type}/NormFiltered/{str(normal_df["ecg_id"][i]).zfill(5)}.mat')['ECG'][:, :4000], dtype=torch.float32)[None, :, :])

    for col in hyper_params['visualization_params']['abnorm_labels']:

        labeled_df = df.loc[(
            (df['NORM'] == 0) &
            (df[col] == 1)
        )].reset_index(drop=True)

        for i in range(len(labeled_df)):
            abnormal_ECGs.append(
                torch.as_tensor(
                    scipy.io.loadmat(
                        f'./ECG_embedding_classification/Data/{data_type}/NormFiltered/{str(labeled_df["ecg_id"][i]).zfill(5)}.mat')['ECG'][:, :4000], dtype=torch.float32)[None, :, :])

    normal_embeddings = []
    abnormal_embeddings = []

    for ecg in normal_ECGs:
        normal_embeddings.append(torch.squeeze(
            embedding_model(ecg)).detach().numpy())

    for ecg in abnormal_ECGs:
        abnormal_embeddings.append(torch.squeeze(
            embedding_model(ecg)).detach().numpy())

    if dim_reduction_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(abnormal_embeddings + normal_embeddings)
        tf_normal_embeds = pca.transform(normal_embeddings)
        tf_abnormal_embeds = pca.transform(abnormal_embeddings)
    elif dim_reduction_method == 'UMAP':
        import umap
        fit = umap.UMAP()
        fit.fit(abnormal_embeddings + normal_embeddings)
        tf_normal_embeds = fit.transform(normal_embeddings)
        tf_abnormal_embeds = fit.transform(abnormal_embeddings)

    plt.scatter(
        tf_normal_embeds[:, 0],
        tf_normal_embeds[:, 1],
        label='normal')
    plt.scatter(
        tf_abnormal_embeds[:, 0],
        tf_abnormal_embeds[:, 1],
        label='abnormal')

    plt.legend()
    plt.show()


hyper_params = {
    'visualization_params': {
        'abnorm_labels': ['STTC', 'MI', 'HYP', 'CD'],
        'pca': 'PCA',
        'umap': 'UMAP',
        'test': 'Test',
        'val': 'Val',
        'train': 'Train'
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

    red_method = 'umap'
    data_type = 'test'

    opts, args = getopt.getopt(sys.argv[1:], 'r:d:', ['reduction-method=', 'data-type='])
    for opt, arg in opts:
        if opt in ('-r', '--reduction-method'):
            red_method = arg
        elif opt in ('-d', '--data-type'):
            data_type = arg

    with torch.no_grad():
        norm_vs_all(
            dim_reduction_method=hyper_params['visualization_params'][red_method],
            data_type=hyper_params['visualization_params'][data_type]
        )
