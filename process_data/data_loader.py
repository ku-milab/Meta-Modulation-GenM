from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from os.path import join

from .data_process import *
# from cluster_coefficient import cluster_coff_from_FC
from .custom_dataset import DatasetCustom
from .custom_sampler import EpisodeSampler

class ABIDE_loader:
    def __init__(self):
        self.base_train_set, self.intra_val_set, self.intra_test_set, self.episodes_ASD_set, self.episodes_TC_set = [], [], [], [], [] # from source dataset
        self.inter_test_set = [] # from unseen dataset

    def setup_source(self, path, fc_type, roi_type, site_list, fold_idx=0, batch_size=8):
        intra_val_d, intra_val_l = [], []
        # intra_val_d, intra_val_l = np.array(intra_val_d), np.array(intra_val_l)
        for i, site in enumerate(site_list):
            dataset_ASD = np.load(join(path, 'ABIDE1_fcv_'+fc_type, roi_type, site+'_'+'ASD.npy'))
            dataset_TC = np.load(join(path, 'ABIDE1_fcv_'+fc_type, roi_type, site+'_'+'TC.npy'))
            ASD_train_idx = np.load(join(path, 'ABIDE1_cv_idx','train_val_test_ratio', site+'_ASD_train_f{}.npy'.format(fold_idx)))
            ASD_val_idx = np.load(join(path, 'ABIDE1_cv_idx','train_val_test_ratio', site+'_ASD_val_f{}.npy'.format(fold_idx)))
            ASD_test_idx = np.load(join(path, 'ABIDE1_cv_idx', site+'_ASD_test_f{}.npy'.format(fold_idx)))
            TC_train_idx = np.load(join(path, 'ABIDE1_cv_idx', site+'_TC_train_f{}.npy'.format(fold_idx)))
            TC_val_idx = np.load(join(path, 'ABIDE1_cv_idx','train_val_test_ratio', site+'_TC_val_f{}.npy'.format(fold_idx)))
            TC_test_idx = np.load(join(path, 'ABIDE1_cv_idx', site+'_TC_test_f{}.npy'.format(fold_idx)))

            base_train_set = np.concatenate((dataset_ASD[ASD_train_idx], dataset_TC[TC_train_idx]), axis=0)
            base_train_label = np.concatenate((np.ones(len(dataset_ASD[ASD_train_idx]), dtype=np.int), np.zeros(len(dataset_TC[TC_train_idx]), dtype=np.int)))

            base_train = DatasetCustom(torch.from_numpy(base_train_set), torch.from_numpy(base_train_label))
            self.base_train_set.append(DataLoader(base_train, num_workers=1, batch_size=batch_size,shuffle=True,pin_memory=True,drop_last=False))

            intra_set = np.concatenate((dataset_ASD[ASD_test_idx], dataset_TC[TC_test_idx]), axis=0)
            intra_label = np.concatenate((np.ones(len(dataset_ASD[ASD_test_idx]), dtype=np.int), np.zeros(len(dataset_TC[TC_test_idx]), dtype=np.int)))
            self.intra_test_set.append([torch.from_numpy(intra_set), torch.from_numpy(intra_label)])

            self.episodes_ASD_set.append([torch.from_numpy(dataset_ASD[ASD_train_idx]), torch.from_numpy(np.ones(len(dataset_ASD[ASD_train_idx]), dtype=np.int))])
            self.episodes_TC_set.append([torch.from_numpy(dataset_TC[TC_train_idx]), torch.from_numpy(np.zeros(len(dataset_TC[TC_train_idx]), dtype=np.int))])

            intra_val_d.extend(dataset_ASD[ASD_val_idx])
            intra_val_d.extend(dataset_TC[TC_val_idx])
            intra_val_l.extend(np.ones(len(dataset_ASD[ASD_val_idx]), dtype=np.int))
            intra_val_l.extend(np.zeros(len(dataset_TC[TC_val_idx]), dtype=np.int))

        intra_val_d = np.array(intra_val_d)
        intra_val_l = np.array(intra_val_l)
        self.intra_val_set = [torch.from_numpy(intra_val_d), torch.from_numpy(intra_val_l)]

        return self.base_train_set, self.intra_val_set, self.intra_test_set, self.episodes_ASD_set, self.episodes_TC_set

    def setup_unseen(self, path, fc_type, roi_type, site_list):
        for i, site in enumerate(site_list):
            dataset_ASD = np.load(join(path, 'ABIDE1_fcv_' + fc_type, roi_type, site + '_' + 'ASD.npy'))
            dataset_TC = np.load(join(path, 'ABIDE1_fcv_' + fc_type, roi_type, site + '_' + 'TC.npy'))

            inter_test_set = np.concatenate((dataset_ASD, dataset_TC), axis=0)
            inter_test_label = np.concatenate((np.ones(len(dataset_ASD), dtype=np.int), np.zeros(len(dataset_TC), dtype=np.int)))
            self.inter_test_set.append([torch.from_numpy(inter_test_set), torch.from_numpy(inter_test_label)])
        return self.inter_test_set

class MultiDomain:
    def load_onedomain(self, path, site, fold_idx=0, batch_size=8, sw=False):
        datalist = load_mat2np(join(path, site), 'ROI')
        labels = np.load(join(path, site, 'label.npy'))
        labels[np.where(labels==2)]=0
        train_idx = np.load(join(path, 'fold_idx', '{}_train_f{}.npy'.format(site, fold_idx)))
        val_idx = np.load(join(path, 'fold_idx', '{}_val_f{}.npy'.format(site, fold_idx)))
        test_idx = np.load(join(path, 'fold_idx', '{}_test_f{}.npy'.format(site, fold_idx)))

        if sw:
            train_list, train_label, _ = Sliding_window(datalist[train_idx], labels[train_idx], 30, 1)
            val_list, val_label, _ = Sliding_window(datalist[val_idx], labels[val_idx], 30, 1)
            test_list, test_label, trunc = Sliding_window(datalist[test_idx], labels[test_idx], 30, 1)
        else:
            train_list, train_label = datalist[train_idx], labels[train_idx]
            val_list, val_label = datalist[val_idx], labels[val_idx]
            test_list, test_label, trunc = datalist[test_idx], labels[test_idx], 1

        train_list = FC(train_list)
        val_list = FC(val_list)
        test_list = FC(test_list)

        train_dataset = DatasetCustom(torch.from_numpy(train_list), torch.from_numpy(train_label))
        self.trainset = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.valset = [torch.from_numpy(val_list), torch.from_numpy(val_label)]
        self.testset = [torch.from_numpy(test_list), torch.from_numpy(test_label)]
        self.truncsize = trunc

        return self.trainset, self.valset, self.testset, self.truncsize



class Datacontainer: # for multi-domain setting sw & FC
    def __init__(self):
        self.source_trainset, self.source_valset, self.source_testset, self.source_truncate_size = [], [], [], []
        self.unseen_testset,  self.unseen_truncate_size = [], []
        self.episode_trainset = []

    def source_set(self, path, sites, fold_idx=0, batch_size=8):
        '''
        load source dataset to train and validation
        :param path:
        :param sites:
        :param fold_idx:
        :param batch_size:
        :return: train loader list and valid loader list of source sites
        '''

        for i, site in enumerate(sites):
            datalist = load_mat2np(join(path, site), 'ROI')
            labels = np.load(join(path, site, 'label.npy'))
            labels[np.where(labels==2)]=0

            train_idx = np.load(join(path, 'fold_idx', '{}_train_f{}.npy'.format(site, fold_idx)))
            val_idx = np.load(join(path, 'fold_idx', '{}_val_f{}.npy'.format(site, fold_idx)))
            test_idx = np.load(join(path, 'fold_idx', '{}_test_f{}.npy'.format(site, fold_idx)))

            # train_data, train_label, train_trunc = Sliding_window(datalist[np.concatenate((train_idx, val_idx))], labels[np.concatenate((train_idx, val_idx))], 30, 1)
            train_list, train_label, train_trunc = Sliding_window(datalist[train_idx], labels[train_idx], 30, 1)
            val_list, val_label, val_turnc = Sliding_window(datalist[val_idx], labels[val_idx], 30, 1)
            test_list, test_label, test_trunc = Sliding_window(datalist[test_idx], labels[test_idx], 30, 1)

            train_list = FC(train_list)
            val_list = FC(val_list)
            test_list = FC(test_list)

            train_dataset = DatasetCustom(torch.from_numpy(train_list),torch.from_numpy(train_label))
            self.source_trainset.append(DataLoader(train_dataset, num_workers=1, batch_size=batch_size,shuffle=True,pin_memory=True,drop_last=False))
            self.source_valset.append([torch.from_numpy(val_list), torch.from_numpy(val_label)])
            self.source_testset.append([torch.from_numpy(test_list), torch.from_numpy(test_label)])
            self.source_truncate_size.append(test_trunc)

        return self.source_trainset, self.source_valset, self.source_testset, self.source_truncate_size

    def episode_set(self, path, sites, fold_idx=0, class_num=2, shot=5):
        for i, site in enumerate(sites):
            datalist = load_mat2np(join(path, site), 'ROI')
            labels = np.load(join(path, site, 'label.npy'))
            labels[np.where(labels==2)]=0

            train_idx = np.load(join(path, 'fold_idx', '{}_train_f{}.npy'.format(site, fold_idx)))
            # val_idx = np.load(join(path, 'fold_idx', '{}_val_f{}.npy'.format(site, fold_idx)))
            # test_idx = np.load(join(path, 'fold_idx', '{}_test_f{}.npy'.format(site, fold_idx)))

            train_list, train_label, train_trunc = Sliding_window(datalist[train_idx], labels[train_idx], 30, 1)
            # val_list, val_label, val_turnc = Sliding_window(datalist[val_idx], labels[val_idx], 30, 1)
            # test_list, test_label, test_trunc = Sliding_window(datalist[test_idx], labels[test_idx], 30, 1)

            train_list = FC(train_list)
            # val_list = FC(val_list)
            # test_list = FC(test_list)

            train_dataset = DatasetCustom(torch.from_numpy(train_list), torch.from_numpy(train_label))
            self.episode_trainset.append(DataLoader(train_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=False, drop_last=False, sampler=EpisodeSampler(train_dataset, class_num=class_num, class_sample_size=shot, set_len=100)))
        return self.episode_trainset

    def unseen_set(self, path, sites):
        for i, site in enumerate(sites):
            datalist = load_mat2np(join(path, site), 'ROI')
            labels = np.load(join(path, site, 'label.npy'))
            labels[np.where(labels==2)]=0

            sw_data, sw_label, trunc = Sliding_window(datalist, labels, 30, 1)
            datalist = FC(sw_data)
            self.unseen_testset.append([torch.from_numpy(datalist), torch.from_numpy(sw_label)])
            self.unseen_truncate_size.append(trunc)

        return self.unseen_testset, self.unseen_truncate_size



class AGGDatacontainer():
    """Aggregated dataset of multi-site"""
    def __init__(self):
        self.trainset, self.validset, self.testset = [],[],[]

    def data_list(self, path, sites, fold_idx=0, batch_size=8):
        for i, site in enumerate(sites):
            datalist = load_mat2np(join(path, site), 'ROI')
            datalist = FC(datalist)
            labels = np.load(join(path, site, 'label.npy'))

            train_idx = np.load(join(path, 'fold_idx', '{}_train_f{}.npy'.format(site, fold_idx)))
            val_idx = np.load(join(path, 'fold_idx', '{}_val_f{}.npy'.format(site, fold_idx)))
            test_idx = np.load(join(path, 'fold_idx', '{}_test_f{}.npy'.format(site, fold_idx)))

            train_dataset = DatasetCustom(datalist[train_idx], labels[train_idx])
            self.trainset.append(DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False))
            self.validset.append([datalist[val_idx],labels[val_idx]])
            self.testset.append([datalist[test_idx],labels[test_idx]])

        return self.trainset, self.validset, self.testset