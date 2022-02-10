#!/bin/bash


for fold in 0 1 2 3 4
do

python run_main.py \
--exp "exp_1" \
--fold $fold \
--root_path "." \
--fc_type "correlation" \
--roi_type "AAL1_116" \
--input_dim 6670 \
--hidden_dim1 8 \
--weights_init "He" \
--feature_activation "gelu" \
--modulation_method "self_att" \
--batch_size 8 \
--shot 4 \
--lr 1e-4 \
--meta_learning_steps 10000 \
--base_steps 1 \
--episode_iter_steps 1 \
--meta_train_steps 3 \
--meta_test_steps 2 \
--gen_train_steps 10000
done
