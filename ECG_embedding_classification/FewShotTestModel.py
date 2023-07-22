import os
import sys
import json
import torch
import getopt
from typing import List
import matplotlib.pyplot as plt
from Models.EmbeddingModel import EmbeddingModule
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from Datasets.FewShotDataset import FewShotDataset


def show_results(metrics_type: str, labels: List[str]):
    with open(f'ECG_embedding_classification/ModelMetrics/Norm_vs_{labels}.json') as json_file:
        data = json.load(json_file)
    plt.plot(data['shots'], data[metrics_type], label=metrics_type)
    plt.title(f'Normal vs {labels}')
    plt.xlabel('SHOTS')
    plt.ylabel(metrics_type)
    plt.legend()
    plt.show()


def train(
    model: EmbeddingModule,
    dataset: FewShotDataset,
    n_neigh: int
):

    X = []
    y = []
    with torch.no_grad():
        for i in range(dataset.get_train_size()):
            ecg, label = dataset.get_train_item(i)
            X.append(
                torch.squeeze(model(ecg)).detach().numpy()
            )
            y.append(label)

    classifier = KNeighborsClassifier(n_neighbors=n_neigh)
    classifier.fit(X, y)

    return classifier


def test(
    model: EmbeddingModule,
    classifier: KNeighborsClassifier,
    dataset: FewShotDataset
):

    X = []
    y = []
    with torch.no_grad():
        for i in range(dataset.get_test_size()):
            ecg, label = dataset.get_test_item(i)
            X.append(
                torch.squeeze(model(ecg)).detach().numpy()
            )
            y.append(label)

    predictions = []
    for i in range(len(X)):
        predictions.append(
            classifier.predict(X[i].reshape(1, -1))
        )

    return accuracy_score(y_true=y, y_pred=predictions), f1_score(
        y_true=y, y_pred=predictions, pos_label=1, average='binary')


def test_norm_vs_all(
    shots: List[int],
    neighbours: List[int],
    labels: List[str],
    metrics_type: str
):

    if not os.path.exists('ECG_embedding_classification/ModelMetrics/'):
        os.mkdir('ECG_embedding_classification/ModelMetrics')
    if not os.path.exists(
            f'ECG_embedding_classification/ModelMetrics/Norm_vs_{labels}.json'):

        history = {
            'shots': [0],
            'accuracy': [0.0],
            'f1_score': [0.0]
        }

        for shot in shots:
            dataset = FewShotDataset(abnormal_labels=labels, shot=shot)

            b_acc = -float('inf')
            b_f1 = -float('inf')

            for neigh in neighbours:

                if float.__ceil__(shot / 2) < neigh:
                    continue

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

                classifier = train(
                    embedding_model, dataset=dataset, n_neigh=neigh)
                acc, f1 = test(embedding_model, classifier, dataset=dataset)

                if b_acc < acc:
                    b_acc = acc
                if b_f1 < f1:
                    b_f1 = f1

                print(f'shot: {shot} | neighb: {neigh} | accuracy: {acc}, f1_score: {f1}')

            history['shots'].append(shot)
            history['accuracy'].append(b_acc)
            history['f1_score'].append(b_f1)

        with open(f'ECG_embedding_classification/ModelMetrics/Norm_vs_{labels}.json', 'w') as history_file:
            history_file.write(json.dumps(history))

    show_results(
        metrics_type=metrics_type,
        labels=labels
    )


hyper_params = {
    'test_params': {
        'shots': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'neighbours': [1, 3, 5],
        'accuracy': 'accuracy',
        'f1': 'f1_score',
        'all': ['STTC', 'MI', 'HYP', 'CD'],
        'st': ['STTC'],
        'mi': ['MI']
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

    labels = 'all'
    metrics = 'accuracy'

    opts, args = getopt.getopt(sys.argv[1:], 'l:m:', ['labels=', 'metrics='])
    for opt, arg in opts:
        if opt in ('-l', '--labels'):
            labels = arg
        elif opt in ('-m', '--metrics'):
            metrics = arg

    if labels not in ('all', 'st', 'mi'):
        raise ValueError('\"labels\" option must be in {\'all\', \'st\', \'mi\'}')
    if metrics not in ('accuracy', 'f1'):
        raise ValueError('\"metrics\" option must be in {\'accuracy\', \'f1\'}')

    with torch.no_grad():
        test_norm_vs_all(
            shots=hyper_params['test_params']['shots'],
            neighbours=hyper_params['test_params']['neighbours'],
            labels=hyper_params['test_params'][labels],
            metrics_type=hyper_params['test_params'][metrics]
        )
