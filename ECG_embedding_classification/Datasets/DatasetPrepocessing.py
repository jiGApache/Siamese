import os
import re
import ast
import wfdb
import scipy
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
import concurrent.futures

sampling_rate = 500
connections = 50
filter_method = "neurokit"


def load_raw_data(df, path):
    data = [wfdb.rdsamp(f'{path}/{f}') for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dict, agg_df):
    tmp = []
    for key in y_dict.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def load_file(url, file_path, timeout):
    r = requests.get(url=url, timeout=timeout)
    with open(file_path, 'wb') as file:
        file.write(r.content)


def load_data(path):

    os.mkdir(path)
    load_file(url=f'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv',
              file_path=f'{path}/ptbxl_database.csv', timeout=30)
    load_file(url=f'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv',
              file_path=f'{path}/scp_statements.csv', timeout=30)

    os.mkdir(f'{path}/records{sampling_rate}')
    data_root = requests.get(
        f'https://physionet.org/files/ptb-xl/1.0.3/records{sampling_rate}')
    folders = re.findall(r'(?<=>)\d*\/(?=<)', str(data_root.content))

    for folder in folders:
        os.mkdir(f'{path}/records{sampling_rate}/{folder}')

        files = requests.get(
            f'https://physionet.org/files/ptb-xl/1.0.3/records{sampling_rate}/{folder}')
        files = re.findall(r'(?<=>)\d*_hr.[a-z]*(?=<)', str(files.content))

        file_links = [
            f'https://physionet.org/files/ptb-xl/1.0.3/records{sampling_rate}/{folder}/{file}' for file in files]
        file_paths = [
            f'{path}/records{sampling_rate}/{folder}/{file}' for file in files]

        with tqdm(total=len(files), desc=f'loading {folder} folder', ncols=100) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as executor:
                futures = (executor.submit(load_file, url, file_path, 30)
                           for url, file_path in zip(file_links, file_paths))
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def split_data(path, types=['Train', 'Val', 'Test']):

    need_to_split = False
    for type in types:
        if not os.path.exists(f'{path}/{type}/Initial'):
            os.makedirs(f'{path}/{type}/Initial')
        if len(os.listdir(f'{path}/Train/Initial')) == 0:
            need_to_split = need_to_split | True

    if need_to_split:

        # load and convert annotation data
        Y = pd.read_csv(f'{path}/ptbxl_database.csv', index_col='ecg_id')
        # String scp_codes to dict
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(f'{path}/scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(
            aggregate_diagnostic, args=(agg_df,))

        X_data = []
        Y_data = []

        val_fold = 9
        test_fold = 10

        # Train
        X_data.append(X[np.where((Y.strat_fold != test_fold) & (
            Y.strat_fold != val_fold))])
        Y_data.append(pd.DataFrame(Y[((Y.strat_fold != test_fold) & (
            Y.strat_fold != val_fold))].diagnostic_superclass))

        # Val
        X_data.append(X[np.where(Y.strat_fold == val_fold)])
        Y_data.append(pd.DataFrame(
            Y[Y.strat_fold == val_fold].diagnostic_superclass))

        # Test
        X_data.append(X[np.where(Y.strat_fold == test_fold)])
        Y_data.append(pd.DataFrame(
            Y[Y.strat_fold == test_fold].diagnostic_superclass))

        for i, dtype in enumerate(types):
            Y_data[i]['STTC'] = Y_data[i]['diagnostic_superclass'].apply(
                lambda x: 1 if 'STTC' in x else 0)
            Y_data[i]['NORM'] = Y_data[i]['diagnostic_superclass'].apply(
                lambda x: 1 if 'NORM' in x else 0)
            Y_data[i]['MI'] = Y_data[i]['diagnostic_superclass'].apply(
                lambda x: 1 if 'MI' in x else 0)
            Y_data[i]['HYP'] = Y_data[i]['diagnostic_superclass'].apply(
                lambda x: 1 if 'HYP' in x else 0)
            Y_data[i]['CD'] = Y_data[i]['diagnostic_superclass'].apply(
                lambda x: 1 if 'CD' in x else 0)
            Y_data[i] = Y_data[i].drop(['diagnostic_superclass'], axis=1)
            Y_data[i].to_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv')
            for i, ecg in zip(Y_data[i].index, X_data[i]):
                scipy.io.savemat(
                    f'{path}/{dtype}/Initial/{str(i).zfill(5)}.mat', {'ECG': np.transpose(ecg)})

        del X
        del Y
        del X_data
        del Y_data


def prepare_data(path):

    # loading data
    if not os.path.exists(path):
        load_data(path)

    # splitting data to different folders
    data_types = ['Train', 'Val', 'Test']
    for dtype in data_types:
        if not os.path.exists(f'{path}/{dtype}/Initial') \
                or len(os.listdir(f'{path}/{dtype}/Initial')) == 0:
            split_data(path, data_types)
            break

    total_data = []

    # filtering ecgs
    for dtype in data_types:
        local_ref = pd.read_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv', index_col=None)

        if not os.path.exists(f'{path}/{dtype}/Filtered'):
            os.mkdir(f'{path}/{dtype}/Filtered')
        if len(os.listdir(f'{path}/{dtype}/Filtered')) < len(local_ref):
            for i in tqdm(
                    range(
                        len(local_ref)),
                    desc=f'Filtering {dtype} ECG:',
                    ncols=100):
                ecg = scipy.io.loadmat(
                    f'{path}/{dtype}/Initial/{str(local_ref["ecg_id"][i]).zfill(5)}.mat')['ECG']
                ecg = filter_ecg(ecg)
                scipy.io.savemat(
                    f'{path}/{dtype}/Filtered/{str(local_ref["ecg_id"][i]).zfill(5)}.mat', {'ECG': ecg})

                if dtype == 'Train':
                    total_data.append(ecg)
        else:
            if dtype == 'Train':
                for i in tqdm(
                        range(
                            len(local_ref)),
                        desc=f'Reading filtered {dtype} ECG:',
                        ncols=100):
                    ecg = scipy.io.loadmat(
                        f'{path}/{dtype}/Filtered/{str(local_ref["ecg_id"][i]).zfill(5)}.mat')['ECG']
                    total_data.append(ecg)

    print("Filtering done!")

    channel_means, channel_stds = get_channel_means_stds(total_data)
    del total_data

    # normalizing filtered ecgs
    for dtype in data_types:
        local_ref = pd.read_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv')

        if not os.path.exists(f'{path}/{dtype}/NormFiltered'):
            os.mkdir(f'{path}/{dtype}/NormFiltered')
        if len(os.listdir(f'{path}/{dtype}/NormFiltered')) < len(local_ref):
            for i in tqdm(
                    range(
                        len(local_ref)),
                    desc=f'Normalizing filtered ECG in {dtype}:',
                    ncols=100):
                ecg = scipy.io.loadmat(
                    f'{path}/{dtype}/Filtered/{str(local_ref["ecg_id"][i]).zfill(5)}.mat')['ECG']
                for k in range(12):
                    ecg[k] = (ecg[k] - channel_means[k]) / channel_stds[k]
                scipy.io.savemat(
                    f'{path}/{dtype}/NormFiltered/{str(local_ref["ecg_id"][i]).zfill(5)}.mat', {'ECG': ecg})

    print(f"Filtered ECG normaization done!")

    print("Dataset preparation complete!")


def filter_ecg(ecg):
    filtered_ecg = []
    for i in range(12):
        filtered_ecg.append(nk.ecg_clean(ecg[i], sampling_rate=500, method=filter_method))
    return np.asarray(filtered_ecg)


def get_channel_means_stds(total_data):

    ECGs = np.asarray(total_data)
    means = ECGs.mean(axis=(0, 2))
    stds = ECGs.std(axis=(0, 2))

    del ECGs

    return means, stds
