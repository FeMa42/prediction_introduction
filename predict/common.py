import os
import numpy as np
from tqdm import tqdm


def load_data():
    extract_file_d = './extracted_data.npy'
    extract_file_l = './extracted_label.npy'
    if not os.path.isfile(extract_file_d) or not os.path.isfile(extract_file_l):
        data = []
        label = []
        max_seq_len = 0
        for file in tqdm(os.listdir('./sequences/')):
            if file.endswith('inputdata.txt'):
                curr_seq = np.loadtxt(os.path.join('./sequences/', file), delimiter=' ')
                data.append(np.cumsum(curr_seq[:, 0:2], axis=0))
                if curr_seq.shape[0] > max_seq_len:
                    max_seq_len = curr_seq.shape[0]
                target_file = file.replace('inputdata', 'targetdata')
                tar_seq = np.loadtxt(os.path.join('./sequences/', target_file), delimiter=' ')
                tmp_label = tar_seq[0, 0:10]
                tmp_label = np.where(tmp_label==1)[0]
                label.append(tmp_label)
        for i in range(len(data)):
            if data[i].shape[0] < max_seq_len:
                data[i] = np.concatenate(
                    [data[i], np.array([data[i][-1, :].tolist(), ] * (max_seq_len - data[i].shape[0]))], axis=0)
        data = np.array(data)
        label = np.array(label)
        np.save(extract_file_d, data)
        np.save(extract_file_l, label)
    else:
        data = np.load(extract_file_d)
        label = np.load(extract_file_l)

    return data, label