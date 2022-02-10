import torch
import math
import numpy as np
from torch.utils.data.sampler import Sampler

# class SamplerTest(Sampler):
#     def __init__(self):
#
#
# class EpisodeSampler(Sampler):
#     def __init__(self):



class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

# #################################
class RandomSampler(Sampler):
    r"""Samples elements randomly with replacement (if iterations > data set).

    Arguments:
        data_source (Dataset): dataset to sample from
        iterations (int): number of samples to return on each call to __iter__
        batch_size (int): number of samples in each batch
    """

    def __init__(self, data_source, iterations, batch_size):
        self.data_source = data_source
        self.iterations = iterations
        self.batch_size = batch_size

    def __iter__(self):
        if self.data_source._train:
            idx = torch.randperm(self.iterations * self.batch_size) % len(self.data_source)
        else:
            idx = torch.randperm(len(self.data_source))
        return iter(idx.tolist())

    def __len__(self):  # pylint: disable=protected-access
        return self.iterations * self.batch_size if self.data_source._train else len(self.data_source)

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class EpisodeSampler(Sampler):

    def __init__(self, dataset, class_num, class_sample_size, set_len):
        # if not isinstance(sampler, Sampler):
        #     raise ValueError("sampler should be an instance of "
        #                      "torch.utils.data.Sampler, but got sampler={}"
        #                      .format(sampler))
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        # if not isinstance(drop_last, bool):
        #     raise ValueError("drop_last should be a boolean value, but got "
        #                      "drop_last={}".format(drop_last))
        # self.sampler = sampler
        # self.data = dataset.data
        self.label = dataset.label
        self.class_num = class_num
        # self.batch_size = batch_size
        # self.drop_last = drop_last
        self.class_sample_size = class_sample_size
        self.set_len = set_len

    def __iter__(self):
        return iter(self.generate_class_indices())

        # batch = []
        # for idx in self.sampler:
        #     batch.append(idx)
        #     if len(batch) == self.batch_size:
        #         yield batch
        #         batch = []
        # if len(batch) > 0 and not self.drop_last:
        #     yield batch

    def generate_class_indices(self):
        idx = torch.arange(len(self.label))
        idx_len = []
        for i in range(self.class_num):
            locals()['idx_c{}'.format(i)] = idx[torch.eq(self.label, i)]
            # locals()['idx_c' + i] = locals()['idx_c' + i][torch.randperm(len(locals()['idx_c' + i]))]
            idx_len.append(len(locals()['idx_c{}'.format(i)]))
        # max_class_len = np.max(idx_len)

        batches = []
        for j in range(self.set_len):
            tmp = []
            tmp = torch.LongTensor(tmp)
            for k in range(self.class_num):
                tmp = torch.cat([tmp,locals()['idx_c{}'.format(k)][torch.randint(len(locals()['idx_c{}'.format(k)]), (self.class_sample_size,))]])
            batches.append(tmp[torch.randperm(len(tmp))].tolist())
        return np.array(batches)

    def __len__(self):
        return self.set_len
        # if self.drop_last:
        #     return len(self.sampler) // self.batch_size
        # else:
        #     return (len(self.sampler) + self.batch_size - 1) // self.batch_size