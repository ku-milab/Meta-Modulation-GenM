import numpy as np
import scipy.io as sio
from glob import glob
import os
from os.path import join
import torch
from nilearn.connectome import ConnectivityMeasure


def load_mat2tensor(path, dataname):
    """
    :param path: parent path of mat files
    :param dataname: head of data in mat file
    :return: data list
    """
    data_list = []
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.mat')]
    file_list = sorted(file_list)
    for i in file_list:
        data = sio.loadmat(join(path, i))
        data = data[dataname]
        data_list.append(torch.from_numpy(data))
    return data_list

def load_mat2np(path, dataname):
    """
    :param path: parent path of mat files
    :param dataname: head of data in mat file
    :return: data list
    """
    data_list = []
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.mat')]
    file_list = sorted(file_list)
    for i in file_list:
        data = sio.loadmat(join(path, i))
        data = data[dataname]
        data_list.append(data)
    return np.array(data_list)
            # np.array(data_list)

def nplist2array(path):
    """
    :param path: parent path of mat files
    :return: data list
    """
    data_list = []
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.npy')]
    file_list = sorted(file_list)
    for i in file_list:
        data = np.load(join(path, i))
        data = np.squeeze(data)
        data_list.append(data)
    return np.array(data_list)
            # np.array(data_list)

def split_list(x, idx):
    tmp = []
    for i in idx:
        tmp.append(x[i])
    return tmp

def upper_tri(input):
    tmp = []
    for i in range(len(input)):
        a = input[i][:]
        tmp.append(a[np.triu_indices_from(a, 1)])
    tmp = np.array(tmp)
    return tmp

def lower_tri(input):
    tmp = []
    for i in range(len(input)):
        a = input[i][:]
        tmp.append(a[np.tril_indices_from(a, -1)])
    tmp = np.array(tmp)
    return tmp

def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt

def FC(input, type='pearson', fisher_t=False):
    ''' type: pearson / partial '''
    tmp = []
    if type == 'pearson':
        for i in range(len(input)):
            a = np.corrcoef(input[i][:], rowvar=False)
            if fisher_t == True:
                a = np.arctanh(a)
            tmp.append(a[np.triu_indices_from(a, 1)])

    if type == 'partial':
        partial = ConnectivityMeasure(kind='partial correlation')
        for i in range(len(input)):
            a = partial.fit_transform([input[i][:]])[0]
            if fisher_t == True:
                a = np.arctanh(a)
            tmp.append(a[np.triu_indices_from(a, 1)])
    tmp = np.array(tmp)
    # tmp = torch.from_numpy(tmp)
    return tmp

def FC_matrix(input, type='pearson', fisher_t=False):
    ''' type: pearson / partial '''
    tmp = []
    if type == 'pearson':
        for i in range(len(input)):
            a = np.corrcoef(input[i][:], rowvar=False)
            if fisher_t == True:
                a = np.arctanh(a)
            tmp.append(a)

    if type == 'partial':
        partial = ConnectivityMeasure(kind='partial correlation')
        for i in range(len(input)):
            a = partial.fit_transform([input[i][:]])[0]
            if fisher_t == True:
                a = np.arctanh(a)
            tmp.append(a)
    tmp = np.array(tmp)
    # tmp = torch.from_numpy(tmp)
    return tmp

def Sliding_window(x, y, win_size, step):
    tmp = []
    tmpl = []
    # tmppred = []

    for n in range(len(x)):  # for subs
        for t in range(0, x.shape[1] + 1, step):  # for time
            if (t + win_size) >= x.shape[1] + 1:
                break
            tmp.append(x[n, t:t + win_size, :])
            tmpl.append(y[n])
            # tmppred.append(x[n, t + win_size, :])  # x data의 t+1을 label로 저장

    x_data = np.array(tmp)
    x_label = np.array(tmpl)
    # x_pred = np.array(tmppred)
    truncated_size = (x.shape[1] - win_size) / step + 1  # truncated data size per one sub T
    truncated_size = int(truncated_size)
    return x_data, x_label, truncated_size


