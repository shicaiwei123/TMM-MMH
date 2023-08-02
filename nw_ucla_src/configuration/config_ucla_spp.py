import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--total_epoch', type=int, default=10)

parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load', default='')  # "/home/CVPR/shicaiwei/pytorch-resnet3d/src/cv/tmp/ckpt_E_136_I_0.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--plot_every', default=10, type=int)

parser.add_argument('--max_iter', default=400000, type=int)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--method', type=str, default='multiKD')

parser.add_argument('--init_mode', type=str, default='kinects',
                    help='the way to init the student net: random, rgb, depth, ir,sp_feature,mse_feature')

parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=2.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--kd_mode', type=str, default='multi_st', help='mode of kd, which can be:'
                                                                    'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                                    'sp/sobolev/cc/lwm/irg/vid/ofd/afd')

parser.add_argument('--data_root', type=str, default="/home/data/shicaiwei/N-UCLA/multiview_action")

parser.add_argument('rgb_pretrain_path', type=str,
                    default="/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/rgb/10/16/ckpt_clip10_E_226_I_1.pth")
parser.add_argument('depth_pretrain_path', type=str,
                    default="/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/depth/10/16/ckpt_clip10_E_308_I_1.pth")
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('weight_patch', type=int, default=1, help='weight for different patch')
parser.add_argument('lr_warmup', type=int, default=0)
parser.add_argument('fc_mode', type=str, default='random', help='random,multi')

args = parser.parse_args()
args.name = args.method + "_" + "_lr_" + str(args.lr) + '_version_' + str(args.version) + '_weight_patch_' + str(
    args.weight_patch)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

split_subject = int(args.rgb_pretrain_path.split('/')[-3])
args.split_subject = split_subject

clip_len = int(args.rgb_pretrain_path.split('/')[-2])
args.clip_len = clip_len
