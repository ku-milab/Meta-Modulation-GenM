import numpy as np
import os
from os.path import join
from glob import glob
import scipy.io as sio
from data_process import load_mat2tensor, load_mat2np, FC, FC_matrix, Sliding_window, connectivity


site_list = ['Caltech','CMU','KKI', 'Leuven','MaxMun','NYU', 'OHSU','Olin', 'Pitt','SBL','SDSU','Stanford','Trinity','UCLA','UM','USM','Yale']
# old_site = ['CALTECH','LEUVEN','NYU','OLIN','SBL','UCLA','USM','OHSU','PITT','SDSU','TRINITY','UM','YALE','STANFORD','CMU','KKI','MAX_MUN']
path = './'
DX = ['ASD','TC']
roi_list = os.listdir(join(path, 'ABIDE1_ROI'))
fc_type = ['covariance', 'correlation', 'partial correlation', 'precision']
exc =[]
for m in fc_type:
    if m == 'correlation':
        fisher=True
    else:
        fisher=False

    for i in site_list:
        for j in roi_list:
            for k in DX:
                if not os.path.isdir(join(path, 'ABIDE1_fcv_' + m, j, i, k)):
                    os.makedirs(join(path, 'ABIDE1_fcv_' + m, j, i, k))

                for n, fn in enumerate(sorted(glob(join(path, 'ABIDE1_ROI', j, i, k, '*')))):
                    filename = fn.split(os.path.sep)
                    sub_id = filename[-1].split('.')

                    sub = sio.loadmat(fn)
                    sub = sub['ROI']

                    sub = np.reshape(sub, ((1,) + sub.shape))

                    fcv = connectivity(sub, type=m, vectorization=True, fisher_t=fisher)
                    try:
                        np.save(join(path, 'ABIDE1_fcv_' + m, j, i, k, sub_id[-2] + '.npy'), fcv)
                        print(fn, 'done!!')
                    except:
                        exc.append(fn)
                        pass
print(exc)



def make_fc(path, sites):
    for i, site in enumerate(sites):
        datalist = load_mat2np(join(path, 'ABIDE1_for_fc', site), 'ROI')
        labels = np.load(join(path, 'ABIDE1_for_fc', site, 'label.npy'))
        labels[np.where(labels == 2)] = 0

        datalist = FC_matrix(datalist)

        np.savez(join(path, 'ABIDE1_fc', site+'_aal_fc_mt.npz'), data=datalist, label=labels)


#
def make_connectivity(path, type='covariance', fisher=False): # for preproc ABIDE1 ROI
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    exc = []
    dirs = path.split(os.path.sep)
    if not os.path.isdir(join(path,'ABIDE1_fcv_' + type, dirs[-2], dirs[-1])):
        os.makedirs(join(path,'ABIDE1_fcv_' + type, dirs[-2], dirs[-1]))

    for i, fn in enumerate(sorted(glob(join(path, '*')))):
        filename = fn.split(os.path.sep)
        sub_id = filename[-1].split('.')

        sub = sio.loadmat(fn)
        sub = sub['ROI']

        sub = np.reshape(sub, ((1,)+sub.shape))

        fcv = connectivity(sub, type=type, vectorization=True, fisher_t=fisher)
        try:
            np.save(join(path,'ABIDE1_fcv_' + type, filename[-3], filename[-2], sub_id[-2]+'.npy'), fcv)
            print(fn, 'done!!')
        except:
            exc.append(fn)
            pass
    print(exc)


