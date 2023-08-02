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
parser.add_argument('--load',
                    default='/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd_max/_ckpt_E_78_I_0.pth')  # "/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd_new/_ckpt_E_38_I_1.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--plot_every', default=10, type=int)
parser.add_argument('--workers', type=int, default=8)

parser.add_argument('--max_iter', default=400000, type=int)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=101)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--method', type=str, default='multiKD_ucf101')

parser.add_argument('--init_mode', type=str, default='kinects',
                    help='the way to init the student net: kinects, rgb, depth, ir,sp_feature,mse_feature')

parser.add_argument('--T', type=float, default=2.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--kd_mode', type=str, default='multi_st', help='mode of kd, which can be:'
                                                                    'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                                    'sp/sobolev/cc/lwm/irg/vid/ofd/afd')

parser.add_argument('--data_root', type=str, default="/home/data/NTUD60")

parser.add_argument('--rgb_pretrain_path', type=str,
                    default="../results/ucf101_0.01/0/rgb/16/ckpt_E_101_I_1.pth")
parser.add_argument('--flow_pretrain_path', type=str,
                    default="../results/ucf101_0.01/0/flow/16/ckpt_E_99_I_1.pth")
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('weight_patch', type=int, default=1, help='weight for different patch')
parser.add_argument('lr_warmup', type=int, default=0)
parser.add_argument('lambda_kd', type=float, default=0.2, help='trade-off parameter for kd loss')

args = parser.parse_args()
args.name = args.method + "_" + "_lr_" + str(args.lr) + '_version_' + str(args.version) + '_init_mode_' + str(
    args.init_mode)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

clip_len = int(args.rgb_pretrain_path.split('/')[-2])
args.clip_len = clip_len
