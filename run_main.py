import argparse
import os
from os.path import join
import sys
import json
import torch
import numpy as np

from solver_feature_modulate import GenM
# from solver_pretrain import Pretrain
# from process_data.extract_features import ExtractFeature

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # set server GPU number when using server GPU

    parser = argparse.ArgumentParser(description='GenM')
    # set configurations
    parser.add_argument('--exp', type=str, default='exp_1')

    parser.add_argument('--save_args', type=bool, default=True)
    parser.add_argument('--load_args', help='args txt file path')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--meta_learning_log', type=str, default='meta_learning_log.csv')
    parser.add_argument('--GenM_train_log', type=str, default='GenM_train_log.csv')
    parser.add_argument('--GenM_test', type=str, default='GenM_test.csv')

    parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--mode', type=str, default='meta_learning', help='meta_learning, GenM_training, intra_inter_test, extract_feat, Pretrain')
    # parser.add_argument('--sw', type=bool, default=False, help='Sliding window of data')

    parser.add_argument('--root_path', type=str, default='.')
    parser.add_argument('--fc_type', type=str, default='correlation', help='correlation, covariance, partial, precision')
    parser.add_argument('--roi_type', type=str, default='AAL1_116', help="[AAL1_116, CC200_200, CC400_392, Dosenbach160_161, HO_110]")
    # parser.add_argument('--data_path', type=str, default='ABIDE1_for_fc')
    parser.add_argument('--feat_path', type=str, default='features')
    parser.add_argument('--sources', default=['NYU','UM','USM','UCLA'], help="['Caltech','CMU','KKI', 'Leuven','MaxMun','NYU', 'OHSU','Olin', 'Pitt','SBL','SDSU','Stanford','Trinity','UCLA','UM','USM','Yale']")
    parser.add_argument('--unseens',
                        default=['Caltech','CMU','KKI','Leuven','MaxMun','OHSU','Olin','Pitt','SBL','SDSU','Stanford','Trinity','Yale'], help="['Caltech','CMU','KKI', 'Leuven','MaxMun','NYU', 'OHSU','Olin', 'Pitt','SBL','SDSU','Stanford','Trinity','UCLA','UM','USM','Yale']")

    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--checkpoint', type=str, default='meta_eval_loss_checkpoint.tar', help='checkpoint file name[meta_eval_loss_checkpoint.tar,meta_test_loss_checkpoint.tar,intra_inter_acc_checkpoint.tar]')
    parser.add_argument('--input_dim', type=int, default=6670, help='fcv dimension')
    parser.add_argument('--hidden_dim1', type=int, default=8)
    # parser.add_argument('--hidden_dim2', type=int, default=8)
    parser.add_argument('--class_num', type=int, default=2)
    # parser.add_argument('--ce_para', type=float, default=0.0)
    parser.add_argument('--weights_init', type=str, default='He', help="['He','xavier']")
    parser.add_argument('--feature_activation', type=str, default='gelu', help='relu, gelu')
    parser.add_argument('--modulation_feature_activation', type=str, default=' ', help='relu, gelu')
    parser.add_argument('--modulation_method', type=str, default='self_att', help='dot_pro, self_att, multihead_att')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--meta_learning_steps', type=int, default=10000)
    parser.add_argument('--base_steps', type=int, default=1)
    parser.add_argument('--episode_iter_steps', type=int, default=1)
    parser.add_argument('--meta_train_steps', type=int, default=3)
    parser.add_argument('--meta_test_steps', type=int, default=2)
    parser.add_argument('--gen_train_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=92)

    parser.add_argument('--optim_scheduler', type=bool, default=False, help='apply learning rate scheduler')

    args = parser.parse_args()

    if args.load_args:
        with open(args.load_args, 'r') as f:
            args.__dict__ = json.load(f)
        args = parser.parse_args()

    if args.save_args==True:
        log_path = os.path.join(args.root_path,args.log_path, args.exp,str(args.fold))
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        with open(join(log_path,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    print(args.device)

    #== set random seed ==#
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(123) # if you are using multi-GPU
    torch.manual_seed(args.seed)
    # random.seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.mode == 'meta_learning':
        solver = GenM(args)
        solver.meta_learning(args)
        solver.GenM_train(args)

    if args.mode == 'GenM_training':
        solver = GenM(args)
        solver.GenM_train(args)

    if args.mode == 'intra_inter_test':
        solver = GenM(args)
        solver.intra_inter_test(args)

    # if args.mode == 'extract_feat':
    #     solver = ExtractFeature(args)
    #     solver.save_feature(args)

    # if args.mode == 'pretrain':
    #     solver = Pretrain(args)
    #     solver.train(args)


if __name__ == "__main__":
    main()
