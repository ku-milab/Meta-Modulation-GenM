import torch
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

class DatasetTest(Dataset):
    def __init__(self):
        self.data = torch.arange(100).view(100, 1).float()
        self.label = torch.zeros(100)
        self.label[:50] = 1

    def __getitem__(self, index):
        # print(index)
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)

class DatasetCustom(Dataset):
    def __init__(self, data_list, labels):
        self.data = data_list
        self.target = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]
        return x, y

class DatasetCustom_multidata(Dataset):
    def __init__(self, data1, data2, data3, labels):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

        self.target = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]
        y = self.target[idx]
        return x1, x2, x3, y

class Dataset4Metatask(Dataset):
    def __init__(self, data_list, labels, seed=123, outer_num=0.3,):
        """ split dataset to inner and outer sets

        Args:
            data_list:
            labels:
            seed:
            outer_num:
        """
        self.data = data_list
        self.target = labels

        data_idx = np.arange(len(self.target))
        inner_idx, outer_idx = train_test_split(data_idx, test_size=outer_num, stratify=self.target, random_state=seed)
        self.inner_data = list(zip(self.data[inner_idx], self.target[inner_idx]))
        self.outer_data = list(zip(self.data[outer_idx], self.target[outer_idx]))

        self._train = True
        self._set_mode()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y

    def train(self):
        self._train = True
        self._set_mode()

    def val(self):
        self._train = False
        self._set_mode()

    def _set_mode(self):
        if self._train:
            self.dataset = self.inner_data
        else:
            self.dataset = self.outer_data




