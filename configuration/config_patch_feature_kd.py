import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=5)

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
parser.add_argument('--method', type=str, default='patch_feature_l3_l4')

parser.add_argument('--teacher_modal', type=str, default='multi', help="the origin modal to train the teacher")
parser.add_argument('--teacher_data', type=str, default='multi',
                    help='data for teacher or the data knowledge you want to transfer:multi,depth,ir')
parser.add_argument('--student_modal', type=str, default='rgb', help='the origin modal to train the student ')
parser.add_argument('--student_data', type=str, default='multi_rgb',
                    help='the data to trained the student modal: multi_rgb,single_rgb')
parser.add_argument('--miss_modal', type=int, default=0)

parser.add_argument('--student_name', type=str, default='resnet18_se', help='the backbone for student: resnet18_se')
parser.add_argument('--init_mode', type=str, default='random',
                    help='the way to init the student net: random, rgb, depth, ir')
parser.add_argument('--dataset', type=str, default='surf')
parser.add_argument("--protocol", type=str, default='race_prot_rdi_4@5')

parser.add_argument('--lambda_kd_logits', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=2.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--kd_mode', type=str, default='st', help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('data_root', type=str,
                    default='/home/shicaiwei/data/liveness_data/CASIA-SUFR')
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('lr_warmup', type=int, default=1)
parser.add_argument('lambda_kd_feature', type=float, default=1.0, help='trade-off parameter for kd loss')

args = parser.parse_args()
args.name = args.dataset + "_" + args.method + "_" + args.teacher_data + "_" + args.student_data + "_lr_" + str(
    args.lr) + '_version_' + str(
    args.version) + '_lambda_' + str(args.lambda_kd_feature)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
