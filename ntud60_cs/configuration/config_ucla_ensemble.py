import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--total_epoch', type=int, default=10)

parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load', default='')  # "/home/CVPR/shicaiwei/pytorch-resnet3d/src/cv/tmp/ckpt_E_136_I_0.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--plot_every', default=10, type=int)

parser.add_argument('--max_iter', default=40000, type=int)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=60)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--method', type=str, default='spp_ensemble_cd')

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--spp_freeze', type=int, default=1)
parser.add_argument('--cd_freeze', type=int, default=1)

parser.add_argument('--data_root', type=str, default="/home/data/NTUD60")

parser.add_argument('spp_pretrained_dir', type=str,
                    default="/home/CVPR/shicaiwei/multi_model_fas/output/models/CDD_ntud60/_ckpt_E_70_I_2.pth")
parser.add_argument('cd_pretrained_dir', type=str,
                    default="/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd/_ckpt_E_70_I_2.pth")

parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('lr_warmup', type=int, default=1)
parser.add_argument('mode', type=str, default='video')

args = parser.parse_args()
args.name = args.method + "_lr_" + str(args.lr) + '_version_' + str(
    args.version)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

args.clip_len = 16
