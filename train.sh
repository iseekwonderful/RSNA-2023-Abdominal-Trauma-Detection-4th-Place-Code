#! /bin/bash

cd organ

python train_net.py train -i ddp_2_maxvit_t_naive_14_3_16_pretrain_207.yaml -j ddp_2/maxvit_t_naive_14_3_16_pretrain_207.yaml --gpu=0123

python train_net.py train -i ddp_2_pvtb2_naive_24_4_14_pretrain.yaml -j ddp_2/pvtb2_naive_24_4_14_pretrain.yaml --gpu=0123

python train_net.py train -i ddp_2_pvtb3_naive_30_3_14_pretrain_1xlr.yaml -j ddp_2/pvtb3_naive_30_3_14_pretrain_1xlr.yaml --gpu=0123

python train_net.py train -i ddp_2_pvtb4_naive_v4_24_2_14_pretrain_step2.yaml -j ddp_2/pvtb4_naive_v4_24_2_14_pretrain_step2.yaml --gpu=0123

cd ../
