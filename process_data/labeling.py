import csv
import os
import numpy as np

def find_csv(csv_path, row_comp_idx, row_re_idx, find_la):
    '''
    Args:
        csv_path: csv file to find suitable condition and return the label
        row_comp_idx: location of condition to find label
        row_re_idx: location of label
        find_la: condition file name

    Returns: label
    '''
    with open(csv_path, 'r', encoding='cp949') as f:
        data = csv.reader(f)
        for row in data:
            if row[row_comp_idx] in find_la:
                return row[row_re_idx]

def labeling_from_csv(file_path,label_file, comp_idx, re_idx):
    """
    Args:
        file_path:
        label_file: csv file to find label
        comp_idx: location of condition to find label
        re_idx: location of label

    Returns: save label file to numpy
    """
    file_list = os.listdir(file_path)
    file_list = [f for f in file_list if f.endswith('.mat')]
    file_list = sorted(file_list)
    print(file_list)

    label = []
    for i in file_list:
        i_label = find_csv(label_file, comp_idx, re_idx, i)
        i_label = int(i_label)
        label.append(i_label)
    if len(file_list) == len(label):
        print('complete!')
    print(label)
    np.save(file_path+'/label', label)

###########################################################################################
site_list = os.listdir('/home/j/Desktop/fMRI_Data/ABIDE/ABIDE1_for_fc')
for site in site_list:
    labeling_from_csv('/home/j/Desktop/fMRI_Data/ABIDE/ABIDE1_for_fc/'+site, '/home/j/Desktop/fMRI_Data/ABIDE/Phenotypic_V1_0b_preprocessed1.csv', 4, 7)