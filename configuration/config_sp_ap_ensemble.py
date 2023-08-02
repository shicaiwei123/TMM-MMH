import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--total_epoch', type=int, default=10)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--method', type=str, default='sp_ensemble_ap_multiKD_acer_best')
parser.add_argument('--sp_model', type=str,
                    default='patch_feature_kd_mmd_avg_sp_multi_multi_rgb_lr_0.001_version_7_sift_0_acer_best_.pth')
parser.add_argument('--sp_freeze', type=int, default=1)
parser.add_argument('--ap_model', type=str,
                    default='patch_kd_multiKD_multi_multi_rgb_lr_0.001_version_4_weight_patch_0_select_0_acer_best_.pth')
parser.add_argument('--ap_freeze', type=int, default=1)
parser.add_argument('--modal', type=str, default='rgb', help='para for initial net')
parser.add_argument('--miss_modal', type=int, default=0, help='para for incomplete modal')
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--kd_mode', type=str, default='st', help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--patch_num', type=int, default=4, help='patch_num x patch_num')
parser.add_argument('--weight_patch', type=bool, default=True, help='weight for different patch')

parser.add_argument('data_root', type=str,
                    default='/home/shicaiwei/data/liveness_data/CASIA-SUFR')
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('lr_warmup', type=int, default=0)

args = parser.parse_args()
args.name = args.method + "_" + "_lr_" + str(args.lr) + '_version_' + str(
    args.version)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.sift = 0
