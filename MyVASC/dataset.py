import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# from scipy.io import mmread

config = {
    'epoch': 2000,
    'min_stop': 500,
    'batch_size': 256,
    'latent': 2,
    'latent_dist': 'Negative binomial',
    'log': False,
    'scale': True,
    'var': False,
    'patience': 50,
    'threshold': 1,
    'annealing': False,
    'anneal_rate': 0.0003,
    'tau0': 1.0,
    'min_tau': 0.5,
    'clustering_method': 'cmeans'
}


class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.int64)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)  # 返回数据的总个数


# preprocessing for gene-cell matrix(row: gene, col: cell)
# dataset: biase.txt, biase_label.txt


def preprocessing_txt(dataset, log, scale):
    # DATASET = 'biase', PREFIX = 'biase'
    filename = dataset + '.txt'
    data = open(filename)
    head = data.readline().rstrip().split()

    label_file = open(dataset + '_label.txt')
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()

    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)

    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])

    expr = []
    for line in data:
        temp = line.rstrip().split()[1:]
        temp = [float(x) for x in temp]
        expr.append(temp)

    expr = np.asarray(expr).T
    n_cell, _ = expr.shape
    if n_cell > 150:
        batch_size = config['batch_size']
    else:
        batch_size = 16

    expr[expr < 0] = 0.0

    if log:
        expr = np.log2(expr + 1)
    if scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])

    return expr, id_map, label, batch_size


def preprocessing_npy(log, scale):
    # expr = mmread(dataset).astype("int32")
    # expr = expr.transpose().todense()
    expr = np.load('breast_T_cell_gene.npy')
    df = pd.read_csv('metadata.csv')
    label = df[df['celltype_major'] == 'T-cells']['celltype_subset'].to_numpy()

    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])

    n_cell, _ = expr.shape
    if n_cell > 150:
        batch_size = config['batch_size']
    else:
        batch_size = 16

    expr[expr < 0] = 0.0

    if log:
        expr = np.log2(expr + 1)
    if scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])

    return expr, id_map, label, batch_size
