import os
import csv
import numpy as np
import scipy.io as sio
# import bct
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
# from nilearn import connectome
from scipy import stats
#from sklearn.feature_selection import SelectFromModel
#from sklearn.linear_model import LassoCV
from sklearn import linear_model
import pickle
# import tensorflow as tf
import shutil
import json
from tempfile import mkdtemp
# from tqdm import tqdm
from joblib import Parallel, delayed
# import depmeas
# Reading and computing the input data
# Selected pipeline
#pipeline = 'cpac'
seed = 123 #123
np.random.seed(seed)
# tf.set_random_seed(seed)

'''input: (subs, sequences, features) 
select features and return selected features idx'''


def feature_selection(matrix, labels, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection
    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """
    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)
    # featureX = matrix
    # featureY = labels
    featureX = matrix.view(-1, matrix.shape[2])
    matrix_ = featureX
    featureY = np.repeat(labels, matrix.shape[1])
    selector = selector.fit(featureX, featureY)
    x_data = selector.transform(matrix_)
    a = np.array(selector.get_support())
    selected_features = np.where(a == 1)[0]
    # print("Number of labeled samples %d" % len(train_ind))
    # print("Number of features selected %d" % x_data.shape[1])
    return x_data, selected_features

def ttest_feature_selection(matrix, labels):
    trainNormal_idx = np.where(labels == 0)[0]
    trainPatient_idx = np.where(labels == 1)[0]
    # trainNormal_idx = np.where(labels[train_ind] == 1)[0]
    # trainPatient_idx = np.where(labels[train_ind] == 2)[0]
    matrix1 = matrix[trainNormal_idx,:].view(-1, matrix.shape[2])
    matrix2 = matrix[trainPatient_idx,:].view(-1, matrix.shape[2])
    tTestResult = stats.ttest_ind(matrix1, matrix2, equal_var=False)  # two tail t-test
    # selectedFeatures = np.where(tTestResult.pvalue < 0.01)[0]
    selectedFeatures = np.where(tTestResult.pvalue < 0.00001)[0]
    ###################################
    # x_data = matrix[..., selectedFeatures]
    # file_path = '/home/jiyeon/DATA2/interplet_GCN_final/ADAI/for_present/dAD_nAD/fmri_featureIdx_%d.npy' % (cv)
    # file_path2 = '/home/jiyeon/DATA2/interplet_GCN_final/ADAI/for_present/dAD_nAD/fmri_featureIdx_%d_p_val.npy' % (cv)
    # directory = os.path.dirname(file_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # sio.savemat(file_path2, {'p_val': tTestResult.pvalue})
    # sio.savemat(file_path, {'selectedFeatures': list(selectedFeatures)})
    # np.save(file_path2,tTestResult.pvalue)
    # np.save(file_path,selectedFeatures)
    return selectedFeatures

def lasso_feature_selection(matrix, labels):
    #clf = linear_model.Lasso(alpha=0.0001)
    #clf = linear_model.Lasso(alpha=0.0003)
    #clf = linear_model.Lasso(alpha=0.0006)
    # clf = linear_model.Lasso(alpha=0.001)
    #clf = linear_model.Lasso(alpha=0.003)
    #clf = linear_model.Lasso(alpha=0.006)
    clf = linear_model.Lasso(alpha=0.01)
    # clf = linear_model.Lasso(alpha=0.03)
    #clf = linear_model.Lasso(alpha=0.06)
    matrix_=matrix.view(-1, matrix.shape[2])
    labels_=np.repeat(labels, matrix.shape[1])
    clf.fit(matrix_, labels_)
    selectedFeaturesIdx = np.where(clf.coef_ != 0)[0]
    # x_data = matrix_[..., selectedFeaturesIdx]
    return selectedFeaturesIdx

def ElasticNet_feature_selection(matrix, labels):
    # regr = ElasticNet(random_state=0, alpha=0.00001)
    # regr = ElasticNet(random_state=0, alpha=0.00003)
    # regr = ElasticNet(random_state=0, alpha=0.00006)
    # regr = ElasticNet(random_state=0, alpha=0.0001)
    # regr = ElasticNet(random_state=0, alpha=0.0003)
    # regr = ElasticNet(random_state=0, alpha=0.0006)
    # regr = ElasticNet(random_state=0, alpha=0.001)
    # regr = ElasticNet(random_state=0, alpha=0.003)
    # regr = ElasticNet(random_state=0, alpha=0.006)
    regr = ElasticNet(random_state=0, alpha=0.02)
    # regr = ElasticNet(random_state=0, alpha=0.03)
    # regr = ElasticNet(random_state=0, alpha=0.06)
    matrix_ = matrix.view(-1, matrix.shape[2])
    labels_ = np.repeat(labels, matrix.shape[1])
    regr.fit(matrix_, labels_)
    selectedFeaturesIdx = np.where(regr.coef_ != 0)[0]
    # x_data = matrix[:, selectedFeaturesIdx]
    return selectedFeaturesIdx